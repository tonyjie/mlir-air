# qwen25_1_5b — lessons learned

## Lesson 1 — RoPE-linearity trick lets us add Qwen2 QKV bias without forking the multi-launch ELF (Phase 2, 2026-04-19)

**Situation**: Qwen2 adds 1-D bias to each Q/K/V projection PRE-RoPE. The
shared `rms_gemms_rope` ELF (`llama3/multi_launch_builder/`) was designed
for bias-free LlamaForCausalLM and emits `q_roped = RoPE(normed @ wq)`.

**Key insight**: RoPE is a per-position 2-D rotation, hence linear:

    RoPE(q + bq) = RoPE(q) + RoPE(bq)

**Implementation**: precompute `bq_roped = RoPE(broadcast(bq, seq_len))`
once per layer at preload time, then add it on the host to the ELF's
`q_roped` output via a `_run_cached` monkey-patch. Same pattern as
`_llm_shared/phase_helpers/headfirst_fa.py`. V's bias is plain
broadcast-add (no RoPE).

**Why**: zero modifications to `llama3/multi_launch_builder/`, easy to
disable per-deployment (only Qwen2 family registers bias for layers).
Cost: ~MB-per-layer precomputed bias storage; negligible runtime overhead.

**How to apply**: any Qwen2-family deployment can reuse `qwen25_bias.py`.
For other "small additive transformation needs to slot into a frozen
multi-launch ELF" cases, check whether the transformation is linear with
some downstream op (RMSNorm, RoPE, MatMul) — if so, host-side post-add can
absorb it.

## Lesson 2 — GEMM tile config must satisfy `N % (tile_n*herd_n) == 0` AND `M ≥ tile_m*herd_m` or kernel SILENTLY produces garbage (Phase 2, 2026-04-19)

**Situation**: Qwen2.5-1.5B has `kv_dim = 256`. Default tile config in
`build_rms_gemms_rope_module` is `tile_n=128, herd_n=4` →
`tile_n*herd_n=512`. K/V GEMM with N=256 < 512 doesn't error — it
silently produces corrupt K (verify-step `corr = 0.03`).

Same with M: Phase 2 first attempt used `seq_len=128` to dodge a separate
BD-allocator issue, but `seq_len < tile_m*herd_m = 64*8 = 512` causes the
GEMM to silently produce garbage (block cosine 0.02).

**Why**: the GEMM IR builder doesn't assert these constraints. Lowering
emits the "right" tile loops but the bounds are wrong, leaving partial
results.

**How to apply**: when configuring a new model, FIRST check both:
- `N % (tile_n * herd_n) == 0` for every GEMM in the multi-launch ELF
- `M ≥ tile_m * herd_m` (and divisible)
If either fails, pick a tile config that satisfies (e.g. Qwen2.5 used
`tile_n=64, herd_n=4 → 256` to fit kv_dim=256) OR pad. Don't trust
"compiles cleanly" as evidence of correctness; ALWAYS verify single-block
cosine.

## Lesson 3 — At Qwen2.5-class shapes (emb_dim=1536), the 6-launch rms_gemms_rope ELF exhausts shim BD pool at seq_len=2048 (Phase 2, 2026-04-19)

**Situation**: `seq_len=2048` + `emb_dim=1536` triggers `Allocator
exhausted available buffer descriptor IDs` at AIE lowering. Each Q/V/RoPE
DMA splits into 2-D pattern `(size=512, stride=768)` because 1536 doesn't
fit a single BD dim, and the 6 stitched launches share shim channel BD
pools cumulatively.

llama3 (emb_dim=2048) and llama32_3b (emb_dim=3072) don't hit this
because both numbers split into BD-friendly dims (e.g., 2048 = 2×1024,
3072 = 3×1024 — single-dim with stride or smaller multi-dim).

**Phase 3 prerequisite**: split the 6-launch `rms_gemms_rope` ELF into
two ELFs (`rms_qkv_only` 4-launch + `rope_qk_only` 2-launch) so each fits
the per-channel BD pool. Predecessor builders exist in
`llama3/multi_launch_builder/superseded/{rms_attn_gemms_multi.py,
rope_qk_multi.py}` — likely reusable.

**How to apply**: any future model with emb_dim that doesn't split
cleanly into 1024-aligned BDs (1536, 2560, 3584, etc.) at seq_len ≥ 2048
will likely hit this same wall. Plan the 2-ELF split upfront in Phase 1
classification.

## Lesson 4 — Padding emb_dim to BD-friendly dim breaks GQA semantics if n_heads changes (Phase 2 follow-up, 2026-04-19)

**Situation**: Tried to dodge LESSON 3's BD blowup by padding emb_dim
1536 → 2048 host-side (zero-pad weights, RMSNorm pre-scale by
sqrt(orig_dim/padded_dim) ≈ 0.866). Both `rms_gemms_rope` and `o_ffn`
ELFs compiled cleanly at seq_len=2048 — confirming the BD theory. But
correctness collapsed to cosine 0.77 (vs >0.99 gate).

**Root cause**: padding emb_dim 1536 → 2048 forced `n_heads = 2048/128
= 16` (was 12). With unchanged `n_kv_heads=2`, GQA `group_size` changed
from 12/2=6 to 16/2=8. The Q-to-KV head assignment is wrong:
- Orig: Q heads 0–5 → KV head 0;  Q heads 6–11 → KV head 1
- Padded (broken): Q heads 0–7 → KV head 0;  Q heads 8–15 → KV head 1
- Q heads 6–7 (orig used KV 1) now route to KV 0. Same kind of mismatch
  for heads 8 onward.

CPU-only sanity check (`_pad_cpu_test.py`, since deleted) confirmed the
math diverges with cosine 0.77 even WITHOUT the NPU in the loop — the
bug is purely in the padding scheme.

**Why**: GQA's `n_heads / n_kv_heads = group_size` invariant is a HARD
constraint. Padding n_heads (the only knob if we hold head_dim=128 and
need padded_emb_dim a multiple of 1024) can only preserve the invariant
if n_kv_heads also pads proportionally. For Qwen2.5: 16/2=8 vs orig
12/2=6 — no integer n_kv_heads gives group=6 with padded n_heads=16.

**Workaround that DOES preserve GQA — IMPLEMENTED 2026-04-19**: pad Q
heads with PHANTOM heads INSERTED at group boundaries — i.e., orig
heads 0–5 stay as padded heads 0–5, then 2 phantom (zero) heads at
padded positions 6–7, then orig heads 6–11 placed at padded positions
8–13, then 2 more phantom heads at 14–15. wq, bq, wo all reindexed.
Implemented in `qwen25_pad.py` (`_gqa_reindex_qhead_axis*`).

**Result**: CPU-only orig-vs-padded cosine = 0.999998 (exact within BF16
RMSNorm-scale noise). NPU Phase 2 cosine at seq_len=2048 = 0.9988
(matches the seq_len=512 unpadded path). Phase 3 unblocked.

## Lesson 5 — Two distinct K-DMA repeat-count walls at hidden_dim ≥ ~8200 (Phase 5, 2026-04-19)

**Situation**: Qwen2.5 hidden_dim=8960 hit two SHIM `push_queue.repeat_count`
> 255 walls in decode:

(a) **Down GEMV K-DMA wall**: K=8960 auto-splits as `(outer=K/32=280,
    inner=32)` because the AIE2P vector width is 32 BF16. Outer 280 > 255 ❌.
    K_max under this auto-split is 32 × 255 = **8160** — anything larger
    fails compile.

(b) **B-input shim DMA wall** (separate, even at smaller K!): the input
    vector `_l3_b` is read `launch_count × (tile_m / m_input)` times per
    GEMV. For Gate/Up at M=hidden_dim=8960 with default
    `(tile_m=8, m_input=4)`: `8960/64 × 2 = 280` reads — exceeds 255. Same
    for LM-head per-partition with M=16384: `16384/64 × 2 = 512` ❌.

**Fix (a)**: Added `k_split` parameter to `matvec.build_module`
(default None, back-compat). When set, `src_sizes=[k_split, k_inner]`
explicitly pre-splits the K-DMA so the lowering doesn't auto-split. Use
`k_split=70` for K=8960 → splits as `(70, 128)` — outer 70 ≤ 255 ✓.
Forwarded to Down via `down_k_split` in `o_gemv_ffn_multi`.

**Fix (b)**: Set `tile_m = m_input = 16` for the K=2048 GEMVs. Inner
`tile_m / m_input = 1` cuts reads by 2×. Combined with the larger tile_m
the launch count also drops, so `launch_count × inner` stays under 255
even for M=16384 LM-head partitions.

**Why these are GENERIC fixes** (not just Qwen2.5):
- `k_split` is needed by ANY model with hidden_dim > 8160 (e.g., Llama-3-8B
  has 14336 — would also need `k_split` like 56 → inner 256).
- `tile_m = m_input` is just a tuning; helps any GEMV where M and K are
  large enough to push the B-DMA fires count near the limit.

**Back-compat for shared infra changes**:
- `matvec.build_module(k_split=None)` produces byte-identical IR to before
  (else-branch preserves old `src_sizes=[k]` path). Verified.
- `o_gemv_ffn_multi.build_o_gemv_ffn_module(down_k_split=None)` same —
  forwards None to matvec, no behavior change.
- llama3, smollm2, llama32_3b decode paths exercise the default
  `k_split=None` path → no regression.

**How to apply**: when bringing up decode for a new model:
1. Compute `launch_count × (tile_m / m_input)` for every GEMV. Must be ≤ 255.
2. If `hidden_dim > 8160`, set `down_k_split` to a value where
   `hidden_dim % k_split == 0` and `k_split ≤ 255` and
   `hidden_dim/k_split ≤ ~1023` (BD inner dim limit).
3. CPU sanity test the matvec.build_module with both default and your
   `k_split` to confirm the IR change is what you expected before going
   to hardware compile (saves ~30 s per iteration).

**How to apply**: when padding emb_dim of a GQA model, NEVER assume
`n_heads = padded_emb_dim / head_dim` directly. Either (a) pad with
GQA-aware reindexing, (b) split the ELF instead, or (c) keep emb_dim
unchanged and use a different BD-pool fix. Verify with a CPU-only
orig-vs-padded transformer-block diff BEFORE wiring to NPU — catches
this kind of structural bug in seconds.
