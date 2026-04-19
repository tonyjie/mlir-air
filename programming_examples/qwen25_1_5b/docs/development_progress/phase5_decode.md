# Phase 5 — Decode performance (Qwen2.5-1.5B)

**Date**: 2026-04-19
**Status**: PASS — 5/5 patterns (with `[PARTIAL]` on attention same as llama3 design).

## Headline result

| Metric | Value |
|---|---|
| Steady-state latency | **216 ms/token** (no CPU verify) |
| Throughput          | **4.6 tok/s** |
| Per-layer rate      | **7.7 ms/layer** (28 layers) |
| Top-1 NPU/CPU match | 5/6 (83%) — > 80% gate ✓ |
| Generated text      | "The capital of France is Paris, and the capital of France" |

## Decode infra wired (3 NEW kernel-level pieces)

1. **`mv_k8960.o`** — Down GEMV at K=hidden_dim=8960. Cloned llama3's
   `_ensure_mv_k8192_o` codepath; same `-Dmatvec_*=dg_matvec_*` and
   `-Dlinalg_fill_*=dg_linalg_fill_*` renames; `-DDIM_M_OUTPUT=2`.

2. **NPU LM Head GEMV with 10 partitions × 16384** — covers vocab=151936
   (10 × 16384 = 163840, padded by 11904). The shared
   `lm_head_gemv_multi.build_lm_head_gemv_module` was already parametric
   on `n_partitions`, just needed `tile_m=16, m_input=16` (see Pattern
   tuning below).

3. **`down_k_split=70` in matvec.py** (NEW additive parameter, back-compat
   default None) — pre-splits the K-DMA dim so the lowering doesn't
   auto-split into a `(outer=K/32=280, inner=32)` BD chain that exceeds the
   AIE2P shim's 255 repeat-count hardware limit. With `k_split=70`, splits
   as `(70, 128)` — outer 70 ≤ 255 ✓.

   **Back-compat verified**: at default `k_split=None`, the IR is byte-
   identical to before our change (else-branch preserves old behavior).
   llama3/smollm2/llama32_3b decode paths (all use K=2048 + K=8192 at
   default config) are unaffected.

## Tile-config friction we hit and fixed

The shim B-input DMA fires `launch_count × (tile_m / m_input)` times
per K=2048-class GEMV. With Qwen2.5's M=8960 (Gate, Up, LM-head
partitions) and default `(tile_m=8, m_input=4)`:

  Gate alone: 8960/(8*8) × 2 = 280  → exceeds 255 ❌
  LM-head alone: 16384/(8*8) × 2 = 512 → exceeds 255 ❌

**Fix**: `tile_m=16, m_input=16` → inner_iter = 1, launch_count cut by 2.
  Gate: 8960/(16*8) × 1 = 70 ✓
  LM-head per partition: 16384/(16*8) × 1 = 128 ✓
  L2 fits: K=1536 × herd=8 × tile_m=16 × 2B = 384KB < 512KB ✓
  L1 fits: m_input=16 × K=1536 × 2B = 48KB < 64KB ✓

## Pattern application

| # | Pattern | Status | Notes |
|---|---|---|---|
| 1 | Multi-launch merging | INHERITED | rms_gemv_rope=6, o_gemv_ffn=8 launches |
| 2 | Static weight BOs | INHERITED | decode preload (`pre_transpose_decode_weights` + per-layer BOs) |
| 3 | **NPU LM Head GEMV** | **APPLIED** | 10×16384 partition (vocab=151936), `tile_m=16, m_input=16` |
| 4 | **Extern kernel rename** | **APPLIED** | `mv_k8960.o` for Down GEMV K=8960; `down_k_split=70` for the K-DMA pre-split |
| 5 | CPU→NPU op promotion | PARTIAL | Attention stays on CPU (llama3 decode design — same for all deployments) |

## Comparison vs prior deployments (steady-state decode)

| Model | n_layers | head_dim | ms/token | tok/s | ms/layer |
|---|---|---|---|---|---|
| llama3 (1B)         | 16 | 64  | 92  | 10.8 | 5.75 |
| smollm2 (1.7B)      | 24 | 64  | 137 | 7.3  | 5.7  |
| llama32_3b          | 28 | 128 | 215 | 4.7  | 7.7  |
| **qwen25_1_5b**     | **28** | **128** | **216** | **4.6** | **7.7** |

**Per-layer rate matches llama32_3b exactly** (7.7 ms/layer) — both at
head_dim=128, n_layers=28. Qwen2.5's smaller emb_dim=1536 (vs llama32_3b's
3072) doesn't dominate decode time at M=1.

## Lessons (added to LESSONS.md)

The `down_k_split` knob is the new reusable insight here — any future model
with `hidden_dim > 8160` will hit the same `repeat_count > 255` wall and
should use this knob.

## Files produced

- `qwen25_decode_setup.py` — `ensure_mv_k8960_o`,
  `compile_qwen25_decode_kernels`, `qwen25_npu_lm_head_gemv`,
  `preload_qwen25_lm_head`.
- `qwen25_phase5_test.py` — Phase 5 driver (CPU prefill seeds KV → NPU
  decode loop with per-layer bias + 10-partition LM head).
- **Modified shared infra** (back-compat verified):
  - `matrix_vector_multiplication/bf16/matvec.py` — added `k_split`
    parameter (default None preserves old behavior).
  - `llama3/multi_launch_builder/o_gemv_ffn_multi.py` — added
    `down_k_split` parameter (default None preserves old behavior).
  - `qwen25_bias.py` — refactored to monkey-patch
    `KernelCache.load_and_run` (covers both prefill `rms_gemms_rope` and
    decode `rms_gemv_rope`); added `set_decode_position(pos)` API.
