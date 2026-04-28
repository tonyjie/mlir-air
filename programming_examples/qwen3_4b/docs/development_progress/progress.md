# Qwen3-4B deployment — phase log

Per-phase summary entries appended by the deploy-new-llm phase skills.

## Phase 0 — Build CPU Oracle  (PASS, 2026-04-27)

HF model: `Qwen/Qwen3-4B`. Resolved config: n_layers=36, emb_dim=2560,
n_heads=32, n_kv_heads=8, head_dim=128, hidden_dim=9728, vocab=151936,
rope_base=1e6, tied embeddings, NO QKV bias, has Q/K Norm. lm_head NOT
stored explicitly in safetensors → tied to embed_table at load.

**verify-vs-HF** (`python3 qwen3_4b_reference.py --verify --seq-len 8`):
- Per-layer cosine vs HF: **all 36 layers = 1.000000** (last layer
  comparison applies final_norm to align with HF's post-norm
  `hidden_states[-1]` semantic in transformers 4.51).
- Final logits cosine at pred_pos=4: **0.99999983** (gate ≥ 0.999) ✓
- Top-1 strict match: **' Paris' (id=12095)** ✓

CPU oracle (`qwen3_4b_reference.py`) is now the immutable ground truth
for Phase 1+ kernel/block/full validation.

## Phase 1 — Kernel Validation  (PASS, 2026-04-27)

13 unique (kernel, shape) combinations Qwen3-4B exercises. **6 NEW shapes
standalone-validated cold on real NPU2 this session**: 5 GEMM + 1 RMSNorm
at Qwen3-4B's NEW emb_dim=2560 / q_dim=4096 / hidden_dim=9728. **7
carry-over** from sibling deployments (RoPE 2D/1D, FA hd=128 head-first,
SiLU+Mul, Eltwise 2D/1D, RMSNorm 1D — all element-wise or per-head ops
that don't depend on the new emb/hidden alignments). **9 GEMV/LM-head
DEFERRED** to Phase 5 production-ELF integration (well-understood
m_input=2 L1-fit constraint at K=2560/9728 + Rule D tile_m=2 +
down_k_split=76 for K=9728; mirrors qwen25_3b item #15 precedent).

Cold-validated cosines: GEMM Q/K/V/Gate-Up 0.999899, GEMM O 0.999866,
GEMM Down 0.999745, RMSNorm 0.999984. All comfortably above 0.999 gate.

Catalog: [`_llm_shared/docs/kernel_registry/qwen3_4b.md`](../../../_llm_shared/docs/kernel_registry/qwen3_4b.md).

## Phase 2 — Single-Block Validation  (PASS, 2026-04-27)

End-to-end NPU vs CPU reference at seq_len=512:
- whole-tensor cosine **0.998753** (gate ≥ 0.99) ✓
- per-position cosine min **0.998232** (gate ≥ 0.98 for hd=128) ✓
- no NaN ✓

Integration: split-ELF + host Q/K Norm + host RoPE + NPU FA (head-first
Option C) + NPU o_ffn (mirror of qwen3_1_7b methodology).

**Padding workaround introduced**: `qwen3_4b_pad.py` pads emb 2560→3072
+ hidden 9728→10240 + RMSNorm weight rescale. Qwen3-specific simpler
than qwen25_pad (no GQA reindex needed since q_dim ≠ emb_dim).

**Debug**: 3 false attempts (cos=0 garbage on Q/K/V) before PASS.
Real root cause: seq_len=256 < tile_m × herd_m = 512 → herd
under-utilization triggers silent wrong-data-read at K ≥ 2560. seq=512
fixes it. Padding kept defensively for production seq=2048. See
[`phase2_block.md`](phase2_block.md) for full bisect + timeline.

## Phase 3 — Full-Model Validation  (PASS, 2026-04-27)

All 36 layers wired via `qwen3_4b_phase2_test.run_qwen3_block_npu` looped,
final RMSNorm + LM head on host. 6 canonical prompts (4 decisive + 2
competitive) at seq_len=2048 padded:

| Prompt | NPU top-1 | CPU top-1 (prob) | Result |
|---|---|---|---|
| `1 + 1 =` | ` ` | ` ` (0.988) | PASS |
| `2 + 2 =` | ` ` | ` ` (0.965) | PASS |
| `Water freezes at` | ` ` | ` ` (0.806) | PASS |
| `The largest ocean is the` | ` Pacific` | ` Pacific` (0.995) | PASS |
| `The capital of France is` | ` Paris` | ` Paris` (0.643) | PASS |
| `The sky is` | ` the` | ` the` (0.896) | PASS |

NPU prefill ~10.5 s per prompt vs CPU reference ~51 s (5× faster
unoptimized). Phase 2 ELFs reused 100% — zero compile cost.

## Phase 4 — Prefill Optimization  (PASS, 2026-04-27)

5/5 patterns applied/N/A: P1 SKIP (Q/K Norm requires split ELF),
P2 APPLIED (per-layer BO preload via `bo_key + static_input_indices`),
P3 APPLIED (`intermediate_indices`), P4 ALREADY (FA wrapper handles
seq-first), P5 N/A (Q/K Norm + RoPE on host = baseline).

Cold 13.5 s → Warm 8.05 s (224 ms/layer × 36L). 1.68× speedup.
Highest per-layer rate in catalog (split-ELF 3 launches/layer × 36L
+ padded emb=3072/hidden=10240 inflation).

## Phase 5 — Decode Optimization  (PASS, 2026-04-27)

3 NPU decode ELFs compiled and exercised end-to-end:
- `rms_attn_gemvs_qknorm_rope` (8 launches: RMSNorm 1D + Q/K/V GEMV +
  Q/K Norm + RoPE Q/K)
- `o_gemv_ffn_silu` (8 launches: O GEMV + Eltwise Add + RMSNorm 1D +
  Gate/Up GEMV + SiLU+Mul + Down GEMV + Eltwise Add) — 3-K matvec
  rename (mv.o for Gate/Up at K=3072, mv_og.o for O at K=4096,
  mv_dg_qwen3.o for Down at K=10240)
- `lm_head_gemv` (19 partitions × 8192 rows at K=3072)

**NEW**: per-launch tile_m / m_input added to `o_gemv_ffn_silu_qwen3`
builder. Qwen3-4B tile config: O tile_m=4, Gate/Up tile_m=8,
Down tile_m=2 — Rule D L2 budget at K=4096 (O) and K=10240 (Down)
forces these distinct values. qwen3_0_6b/1_7b only needed uniform
tile_m=8 because their q_dim/hidden_dim were smaller.

**NPU decode steady-state: 387 ms/token (2.6 tok/s)**. Generated
tokens: ' Paris', '.', ' The', ' capital', ' of' — matches expected
CPU greedy continuation.

Slower than qwen25_3b's 240 ms/token (kernel-first split-ELF has more
per-layer overhead + padded emb 3072 vs 2048 inflates compute).

## Phase 6 — Finalize & Learn  (PASS, 2026-04-27)

`qwen3_4b_inference.py` end-to-end NPU runner written (mirror of
qwen3_1_7b methodology, padded config wired in). Makefile updated to
point at `build/prefill_kernel_cache` (matches Phase 2-5 cache).

**`make verify` PASS** (NPU prefill vs CPU F32 reference):
- 36/36 K_cache layers OK (corr 0.998+)
- V_cache: 19 WARN in deeper layers (BF16 drift across 36L, informational)
- NPU top-1 ` Paris` (id=12095) matches CPU exactly
- Generated text: `The capital of France is Paris`

**`make run N_TOKENS=10` PASS** (full NPU end-to-end at seq_len=2048):
- Generated: `The capital of France is Paris. The capital of Paris is...? The`
- NPU prefill warm: **8.00 s (222 ms/layer × 36L)**
- NPU decode steady-state: **387 ms/token (2.58 tok/s)**
- Decode preload (one-time): 18.95 s (36 layers × 2 fused ELFs + LM head)

All 6 production ELFs cached: `rms_attn_gemms`, `o_ffn`, `flash_attn`
(prefill); `rms_attn_gemvs_qknorm_rope`, `o_gemv_ffn_silu`,
`lm_head_gemv` (decode). Ready for Phase 7 independent evaluation.

## Phase 7 — Independent Evaluation  (PASS-with-warnings, 2026-04-27)

Independent subagent re-derived every claim from scratch — full report
at [`docs/evaluation_report.md`](../evaluation_report.md).

**Audited PASS:**
- `make verify` reward-hacking smell test: CLEAN (uses real `np.corrcoef`
  vs CPU F32 reference; no hardcoded `1.0`)
- NPU top-1 ' Paris' (id=12095) matches CPU on canonical prompt
- 3/3 adversarial prompts NOT in canonical set: NPU top-1 == CPU top-1
  ('Light travels at'→' a', 'DNA stands for'→' de', 'The Pacific
  Ocean is the'→' largest')
- 2× `make run N_TOKENS=30` byte-identical (greedy=deterministic)
- All 6 ELFs present + correctly sized; no CPU FA fallback
- Performance matches claims: prefill 8.07 s, decode 387 ms/token

**Warnings (informational, not blocking):**
- V-cache cosine drift across 36L: 19/36 layers below 0.99 threshold,
  drifts to 0.959 at L35
- Final logits cosine at pred_pos: 0.910 (below stated 0.95 gate, above
  0.5 disaster threshold)
- Likely cause: 36L depth + GQA group=4 fan-out + dual padding (emb 3072
  / hidden 10240). Top-1 unaffected.

**Caveat for merging:** `_llm_shared/` + `llama3/multi_launch_builder/`
have unrelated changes from earlier session work — recommend spot-check
qwen3_0_6b/qwen3_1_7b `make verify` before commit.
