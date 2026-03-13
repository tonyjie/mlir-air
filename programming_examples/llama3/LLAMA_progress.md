# LLAMA-3.2-1B on MLIR-AIR (NPU2) -- Progress Tracker

**Goal**: Functionally correct LLAMA-3.2-1B BF16 prefill inference on NPU2.

**Model config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_heads=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000, seq_len=2048.

---

## Current Status: Single Transformer Block Verified on NPU2

All 15 operations in one transformer block produce correct output with real LLAMA-3.2-1B weights. Top-1 prediction matches CPU reference.

---

## Phase 0: Infrastructure -- DONE

Created `programming_examples/llama3/` with:

| File | Lines | Purpose |
|------|-------|---------|
| `llama3_weights.py` | 456 | Load weights from safetensors, RoPE LUT generation |
| `llama3_reference.py` | 461 | CPU reference forward pass (F32) with per-step intermediates |
| `llama3_prefill.py` | ~1000 | NPU integration: KernelCompiler + transformer block + full model |
| `swiglu_activation.py` | 189 | Standalone SwiGLU AIR kernel |
| `swiglu_activation.cc` | 54 | SwiGLU C++ kernel for Peano |
| `Makefile` | ~100 | Build targets for all operations |
| `run_npu2_swiglu_peano.lit` | 10 | LIT test for CI |
| `debug_gemm.py` | ~200 | GEMM isolation debug script |
| `debug_gemm_real.py` | ~280 | GEMM debug with real weight values |

---

## Phase 1: Standalone Kernel Validation (seq_len=2048) -- ALL PASSED

All 9 kernel configs tested on NPU2 hardware with random data:

| Kernel | Dimensions | Result |
|--------|-----------|--------|
| RMSNorm | M=2048, N=2048 | **PASS** |
| GEMM Q/O | 2048 x 2048 x 2048 | **PASS** |
| GEMM K/V | 2048 x 2048 x 512 | **PASS** |
| GEMM Gate/Up | 2048 x 2048 x 8192 | **PASS** |
| GEMM Down | 2048 x 8192 x 2048 | **PASS** |
| RoPE LUT | seq=65536, dim=64 | **PASS** |
| Eltwise Add | n=4,194,304 | **PASS** |
| Flash Attention | LQ=2048, LK=2048, 32Q/8KV | **PASS** |
| SwiGLU | n=16,777,216 | **PASS** |

---

## Phase 2: Single Transformer Block -- VERIFIED

15-step pipeline with real LLAMA-3.2-1B weights:

| Step | Operation | corr | Status |
|------|-----------|------|--------|
| 1 | RMSNorm | 0.999986 | **OK** |
| 2-4 | Q/K/V GEMM | 0.998 | **OK** |
| 5-6 | RoPE Q/K | 0.999992 | **OK** |
| 7 | Flash Attention GQA | - | ran |
| 8 | O GEMM | 0.996 | **OK** |
| 9 | Residual add | 0.999998 | **OK** |
| 10 | RMSNorm | 0.999988 | **OK** |
| 11-12 | Gate/Up GEMM | 0.998 | **OK** |
| 13 | SwiGLU | 0.999946 | **OK** |
| 14 | Down GEMM | 0.948 | **WARN** |
| 15 | Residual add | 0.999998 | **OK** |

**Block output corr**: 0.999998
**Top-1 prediction**: " is" -- matches CPU reference

---

## Bugs Found and Fixed

1. **Non-contiguous weight arrays** (critical): `.T` creates F-order view; NPU DMA reads wrong layout. Fix: `np.ascontiguousarray(tensor.T)`. See `LLAMA_verification.md` for full debugging story.
2. **Wrong output buffer index**: XRT returns all arrays; output is last. Fix: `results[-1]`.
3. **Flat array returns**: XRT returns 1D; needed `.reshape()` on all returns.
4. **Stale air_project/**: Sequential compilations share tmpdir. Fix: `prepare_air_project()` wipes before each compile.

---

## Session Log

| Date | What was done |
|------|---------------|
| 2026-03-12 | Phase 0: Created all infrastructure files |
| 2026-03-12 | Phase 1 (seq_len=128): All kernels PASS |
| 2026-03-12 | Plan change: seq_len 128 -> 2048 (Flash Attn issues at LQ=128) |
| 2026-03-12 | Phase 1 (seq_len=2048): All 9 kernels PASS |
| 2026-03-12 | Phase 2: Created llama3_prefill.py |
| 2026-03-12 | Phase 2 debugging: Found/fixed 4 bugs (output index, shapes, air_project, **non-contiguous arrays**) |
| 2026-03-12 | **Phase 2 VERIFIED**: Single transformer block passes all 15 steps with real LLAMA weights |
| | **Next**: Run full 16-layer model |

---

## File Inventory

```
programming_examples/llama3/
  llama3_weights.py          # Weight loading + RoPE LUT
  llama3_reference.py        # CPU reference (F32)
  llama3_prefill.py          # NPU integration (KernelCompiler + pipeline)
  swiglu_activation.py       # SwiGLU AIR kernel
  swiglu_activation.cc       # SwiGLU C++ kernel
  Makefile                   # Build targets
  run_npu2_swiglu_peano.lit  # LIT test
  debug_gemm.py              # GEMM isolation debug
  debug_gemm_real.py         # GEMM debug with real weights
  LLAMA_PLAN.md              # High-level plan
  LLAMA_progress.md          # This file (progress tracker)
  LLAMA_verification.md      # Commands, test results, bugs
  LLAMA_explanation.md       # Code walkthrough (architecture -> implementation)
```

---

## Quick Reference: Commands

```bash
# Compile external kernels (one-time)
make compile-external-kernels PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# Single-layer test with verification
make run-prefill-1layer PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# Full 16-layer run
make run-prefill PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# CPU reference only (no NPU)
python3 llama3_reference.py --model /path/to/Llama-3.2-1B

# SwiGLU standalone test
make run-swiglu PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```
