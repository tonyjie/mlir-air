---
name: optimize-decode-perf
description: Phase 5 of LLM deployment — apply the 5 known decode optimization patterns from LLAMA Phase 5 (multi-launch merging, static weight BOs, NPU LM Head GEMV, extern kernel rename, CPU→NPU promotion). Invoked after Phase 4 gate.
---

## Purpose
Apply the decode optimization patterns that took LLAMA from ~500ms/token → 92ms/token. Same structure as Phase 4 but tuned for the M=1 (single-token) case.

## Knowledge base references
- `programming_examples/_llm_shared/docs/multi-launch/decode_merging.md` — decode-specific merge patterns
- `programming_examples/llama3/docs/development_progress/decode_archive/DECODE_PROGRESS.md` — decode milestones
- `programming_examples/llama3/docs/development_progress/decode_archive/gemv_investigation.md` — AIR vs IRON GEMV perf

## Workflow

### Pattern 1: Multi-launch merging (decode variants)
Invoke `merge-multi-launch-kernels` for decode groups (rms+gemvs+rope, o+ffn). Same procedure as prefill but with GEMV instead of GEMM.

Expected: 10 launches/layer/token → 2–3 launches/layer/token.

**Inheritance vs kernel-first** (ADDED 2026-04-21, qwen3-0.6B): if Phase 1
selected the kernel-first path (the model has a new op or an op landing
between currently-fused launches), you'll already have NEW model-specific
fused ELFs from the integration phase. The merging here becomes "make sure
they cover everything": the qwen3 prototype has `rms_attn_gemvs_qknorm_rope`
(8 launches) + `o_gemv_ffn_silu` (8 launches with the **N-way matvec rename**,
see Pattern 4 below) + `lm_head_gemv` = 3 decode ELFs total / 57 NPU calls
per token. The host-side optimizations (Pattern 2 + arg cache + preload, see
qwen3_decode.py) gave a **10× wall speedup** on top of the fusion — fusion
alone was ~unchanged because per-call XRT overhead dominated.

### Pattern 2: Static weight BOs
Decode reuses every weight on every token. Convert weight BOs to allocated-once with `bo.map()` zero-copy access.

Expected gain: removes per-token BO write of all weights.

### Pattern 3: NPU LM Head GEMV (vocab projection)
Replace CPU LM head with NPU GEMV partitioned across vocab (LLAMA used 8-partition with `mv_k8192.o` extern rename). Each partition handles `vocab/8` rows × `emb_dim` columns.

Expected gain: LLAMA observed ~250ms → ~14ms.

### Pattern 4: Extern kernel rename (shape-collision avoidance)
If two GEMV shapes coexist in one ELF (e.g., K=2048 and K=8192) and need different kernel implementations, compile with `-D` symbol renames so they can be linked together.

See `_llm_shared/kernel_builder/external_kernels.py` and `programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py` for the existing K=8192 rename pattern.

**N-way rename for 3+ K shapes** (ADDED 2026-04-21, qwen3-0.6B): the
existing pattern handles 2 K values (default `mv.o` + renamed
`mv_k8192.o` for Down). Qwen3 needed THREE K values in one fused ELF
(O at K=q_dim=2048, Gate/Up at K=emb_dim=1024, Down at K=hidden_dim=3072)
because `n_heads*head_dim != emb_dim`. Solution: extend the rename to
N way by adding more `compile_mv_<group>` helpers that produce
`@<group>_matvec_*` symbols. The qwen3 deployment added:
  - `compile_mv_og(tile_m=8)` → `mv_og.o` exporting `og_matvec_*` for O
  - `compile_mv_dg_qwen3(tile_m=8)` → `mv_dg_qwen3.o` exporting `dg_matvec_*`
    for Down at the qwen3 down_tile_m=8 (the existing `mv_k8192.o` is
    DIM_M_OUTPUT=2 for llama3's down_tile_m=2 — incompatible)

All three .o files compile from the same `mv.cc` with different `-D`
defines; only the exported symbol name differs. The fused-ELF builder
then routes each launch to the right extern via `_rename_all_with_externs`
with per-launch `_EXTERN_<group>` allowlists. See
`programming_examples/qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py`
for the working 3-K example.

**At hidden_dim > 8160, you also need `k_split`** (LESSON 5 from
qwen25_1_5b deployment, 2026-04-19). The Down GEMV's K-DMA auto-splits
as `(outer = K/32, inner = 32)` because the AIE2P BF16 vector width is
32. K_max under auto-split = 32 × 255 = **8160** — the AIE2P shim's
`repeat_count` hardware limit. For K > 8160, pass `down_k_split=N` to
`build_o_gemv_ffn_module` where `K % N == 0` AND `N ≤ 255` AND
`K/N ≤ 1023` (BD inner dim limit). Examples:
- Qwen2.5 K=8960: `down_k_split=70` → splits as (70, 128) ✓
- Llama-3-8B K=14336: `down_k_split=56` → splits as (56, 256) ✓ (proposed)

The `k_split` parameter was added to `matvec.build_module` (default None,
back-compat preserved — verified llama3 K=8192 IR unchanged). Same
mechanism is exposed via `down_k_split` in `o_gemv_ffn_multi`.

**Also at large M (e.g., LM-head partitions M=16384, Gate/Up at M ≥ 8K)**:
the B-input shim DMA fires `launch_count × (tile_m / m_input)` times per
GEMV, and combined GEMVs sharing the channel ADD UP. To stay under 255:
set `tile_m = m_input` (inner_loop=1) and pick `tile_m * herd_m ≥ M / 127`.
Example: Qwen2.5 LM-head per-partition M=16384 → `tile_m=16, m_input=16,
herd_m=8` → `16384/(16*8) × 1 = 128` ✓.

See `programming_examples/_llm_shared/docs/aie2p_hardware_limits.md`
Rules B + C for the full derivation.

### Pattern 5: CPU→NPU op promotion
Same as Phase 4 Pattern 5 but for decode-specific ops (typically the small attention step on a single query, if currently on CPU).

### Bookkeeping
Same as Phase 4: record per-pattern latency in `<model>/docs/development_progress/phase5_decode.md`.

## Verification (Phase 5 gate)

Phase 5 PASSES when:
- Decode latency measured (ms/token)
- ≥3 of 5 patterns applied or N/A
- No correctness regression (Phase 3 re-run still PASS)

## Failure modes
Same as Phase 4 plus:
- Extern kernel rename collision → check `-D` symbol mapping uniqueness; invoke `debug-multi-launch-merge`

## Update protocol
Append to `progress.md`. Update `TODO.md` Phase 5.
