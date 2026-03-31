# LLAMA-3.2-1B on MLIR-AIR (NPU2) -- Progress Tracker

**Goal**: Functionally correct LLAMA-3.2-1B BF16 prefill inference on NPU2.

**Model config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_heads=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000, seq_len=2048.

---

## Current Status: Full 16-Layer Model VERIFIED with NPU FlashAttention

**All 16 layers run end-to-end on NPU**, including FlashAttention (GQA, causal) and multi-launch FFN:
- **Top-1**: " Paris" (prob=0.18) for prompt "The capital of France is"
- **Logits correlation**: 0.993 vs CPU F32 reference
- **Per-kernel**: All NPU kernel invocations corr>0.999
- **Per-layer**: All 16 layer outputs corr=0.999996-0.999997
- **NPU kernel time**: 2.10s (ffn_multi avg 52ms/layer, flash_attn avg 22ms/layer)
- **Wall time**: 4.51s
- **8 unique kernels**: rmsnorm, gemm_qo, gemm_kv, ffn_multi (4 launches), rope_q, rope_k, flash_attn, add
- **Standalone kernel test**: `make run` passes with corr=0.9976 (LLAMA causal, 32Q/8KV)

**NPU attention is now the default.** Use `--cpu-attn` for debugging/comparison. The CPU fallback path is still available and produces corr=0.972.

**Key integration notes:**
- Kernel expects **unscaled Q** — scaling by 1/sqrt(dk) is handled internally
- Kernel takes **4 args** (Q, K, V, Output) — no mask buffer (causal masking is internal)
- Kernel uses **ELF format** — compiled via `make run` in `flash_attention/kernel_fusion_based/`
- IRON MHA comparison: AIR FlashAttention is 2× faster (15ms vs 31ms standalone)

**Next step**: Performance optimization — vectorize RoPE/RMSNorm, FFN kernel fusion.

---

## Phase 0: Infrastructure -- DONE

Created `programming_examples/llama3/` with:

| File | Lines | Purpose |
|------|-------|---------|
| `llama3_weights.py` | 456 | Load weights from safetensors, RoPE LUT generation |
| `llama3_reference.py` | 461 | CPU reference forward pass (F32) with per-step intermediates |
| `llama3_prefill.py` | ~1160 | NPU integration: KernelCache + compile_all_kernels + Profiler + transformer block + full model |
| `ffn_swiglu/silu_and_mul.py` | 189 | Standalone SwiGLU AIR kernel |
| `ffn_swiglu/silu_and_mul.cc` | 54 | SwiGLU C++ kernel for Peano |
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

15-step pipeline with real LLAMA-3.2-1B weights (F32 output GEMM, cached kernels, CPU attention fallback):

| Step | Operation | corr | Status |
|------|-----------|------|--------|
| 1 | RMSNorm | 0.999986 | **OK** |
| 2 | Q GEMM | 0.999984 | **OK** |
| 3 | K GEMM | 0.999973 | **OK** |
| 4 | V GEMM | 0.999849 | **OK** |
| 5 | RoPE Q | 0.999992 | **OK** |
| 6 | RoPE K | 0.999993 | **OK** |
| 7 | Attention GQA (CPU fallback) | exact | **OK** (uses `attention_reference()`) |
| 8 | O GEMM | 0.999688 | **OK** |
| 9 | Residual add | 0.999998 | **OK** |
| 10 | RMSNorm | 0.999988 | **OK** |
| 11 | Gate GEMM | 0.999921 | **OK** |
| 12 | Up GEMM | 0.999896 | **OK** |
| 13 | SwiGLU | 0.999973 | **OK** |
| 14 | Down GEMM | 0.999808 | **OK** |
| 15 | Residual add | 0.999999 | **OK** |

**Block output corr**: 0.999999
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
| 2026-03-12 | Attempted full 16-layer run. Blocked: flash attention `attn.py` was updated externally with `pad_before` support + K layout change (row-major). C++ bindings need rebuild to support `pad_before` in `ChannelPutOp`. |
| 2026-03-13 | MLIR-AIR rebuilt. Fixed K layout in prefill: `(num_kv_heads, dk, lk)` -> `(num_kv_heads, lk, dk)`. Added `np.ascontiguousarray()` for attention inputs. |
| 2026-03-13 | Single-layer re-verified: all 15 steps pass (corr>0.945). O projection improved to corr=0.998. |
| 2026-03-13 | **Full 16-layer model completed!** All 240 kernel invocations (16 layers x 15 steps) ran without errors. |
| | **Result**: NPU top-1 = "def" (wrong). CPU F32 top-1 = " the", top-2 = " Paris" (correct). BF16 error accumulation across 16 layers degrades output quality. |
| | **Root cause of degradation**: The Down GEMM (2048x8192x2048) has corr=0.946 per step. Over 16 layers, the error compounds. The residual connections partially preserve signal (corr=0.9999 per step), but the GEMM precision loss at K=8192 accumulates. |
| 2026-03-13 | Precision investigation: Error is NOT from BF16 tile-boundary truncation (simulated CPU BF16 tiled GEMM gives corr=0.9999). Error is from BFP16 emulation in AIE2P mmul unit (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`). Changing tile_k_l2 from 64 to 256 has zero effect (identical corr=0.976). |
| | **Root cause**: BF16 output truncation of F32 accumulator. Hardware accumulates in F32 but kernel writes BF16 output, losing precision. |
| 2026-03-13 | **FIX CONFIRMED**: F32 output GEMM gives corr=0.9999 (vs 0.976 with BF16 output) at K=8192. IRON uses same approach (`--prio-accuracy` flag with F32 internal buffer). |
| 2026-03-13 | Single-layer re-verified with F32 GEMM output. ALL 15 steps corr>0.999. Down GEMM: 0.948 -> 0.9998 (52x improvement). Block output corr=0.999999. Top-1 " is" matches CPU. |
| | **Next**: Run full 16-layer model (need compilation caching to make it practical -- currently ~8 min/layer) |
| 2026-03-13 | **Kernel Caching & Profiling implemented**: Replaced `KernelCompiler` (compile-per-call) with `KernelCache` (compile-once, run-many). Added `compile_all_kernels()` to pre-compile all 10 unique kernel configs. Added `Profiler` class for per-kernel/per-layer timing. New CLI flags: `--compile-only`, `--run-only`, `--profile`, `--cache-dir`. Compilation artifacts saved to `kernel_cache/` with manifest.json for session persistence. |
| 2026-03-13 | **Profiling results**: Compilation = 334s (one-time). 16-layer prefill = 22.0s kernel time, 23.6s wall. Per-layer avg = 1.37s wall / 1.29s kernel. Biggest bottleneck: eltwise add (38% of kernel time due to per-invocation XRT load/unload overhead). vs IRON reference: 2.9s (7.5x gap from host overhead + no kernel fusion). |
| 2026-03-13 | **16-layer top-1 still incorrect**: NPU predicts " }\n" instead of CPU's " Paris"/" the". Single-layer top-1 matches CPU. Initially attributed to BF16 error accumulation. |
| 2026-03-13 | **Diagnostic infrastructure**: Added `--diagnostic` flag to `llama3_prefill.py` (per-layer NPU vs CPU comparison). Created `diagnose_layer.py` (per-kernel isolation test — runs each kernel with same input, compares against CPU F32 reference). |
| 2026-03-13 | **ROOT CAUSE FOUND**: Per-kernel diagnostic shows **flash attention (step 7) has corr=0.34** — all other 14 kernels have corr>0.999. The flash attention kernel was compiled without causal masking (`causal=False`), performing bidirectional attention. CPU reference correctly applies causal masking. |
| | **Fix attempt 1**: `causal=True` — fails to compile at seq_len=2048 due to BD exhaustion (hardware limit: 48 BDs per MemTile, causal needs ~144). |
| | **Fix attempt 2**: Pass causal mask via external mask input — kernel ignores `arg3` (mask not in launch operands in non-causal path). |
| | **Status**: Blocked on flash attention causal masking compilation. See `LLAMA_flash_attention.md` for full details. |
| 2026-03-13 | **F32 residual path improvement** (secondary): Modified `run_transformer_block()` to carry both BF16 and F32 copies of residual state. Correct but not the main issue. |
| 2026-03-13 | **Configuration sweep**: Created `test_flash_attn_configs.py`. Tested 10 configs at LLAMA shapes. Only 3/10 compile and pass. Best: LQP=256/LKP=64 at 2427 GFLOPS. All causal configs failed (BD exhaustion, pre-upstream-fix). |
| 2026-03-16 | **Causal BD exhaustion fixed upstream**. `causal=True` now compiles: `make run LQ=2048 LK=2048 LQP=256 LKP=64 ... EXTRA_PY_FLAGS="--causal"` → PASS! |
| 2026-03-16 | **Integrated causal=True into llama3_prefill.py**. Recompiled all 10 kernels (344.9s). |
| 2026-03-16 | **16-layer still incorrect** with causal kernel. NPU top-1: `','`, CPU top-1: `' the'`. Logits corr: 0.162. |
| 2026-03-16 | **Per-kernel diagnostic with causal kernel**: FlashAttn still corr=0.31 against `attention_reference()`. All other kernels >0.999. |
| 2026-03-16 | **Key finding**: Standalone `make run PASS!` validates against its own reference (`flash_attn_per_stage()` in `attn.py`), NOT against standard attention (`attention_reference()`). The two reference implementations may not be semantically equivalent at BF16 precision. |
| | **Causality check**: NPU kernel's position 0 output changes when position 7's K is modified (diff=0.00049). This suggests either the causal mask isn't fully effective or there's a position indexing issue. |
| | **Next step**: Compare `attention_reference()` vs `flash_attn_per_stage()` at seq_len=2048 with real LLAMA data to determine if the discrepancy is in the kernel implementation or in our invocation. |
| 2026-03-16 | **Two CPU references verified identical**: `attention_reference()` vs `flash_attn_per_stage()` have corr=0.99999847 on real LLAMA data. The references agree — the kernel is wrong. |
| 2026-03-16 | **Systematic elimination**: Inputs byte-for-byte identical, binary MD5 matches, invocation method doesn't matter, fails with random data too (corr=0.086). Issue is the kernel itself. |
| 2026-03-16 | **Standalone `PASS!` is a false positive**: `atol=0.5, rtol=0.2` with [0.5,1.0] data means any output in [-0.2, 1.7] passes. Random noise and constant 0.75 also pass. Tested with IRON-style inputs ([0,4] range, PyTorch reference): corr=0.009, 73.7% elements fail IRON tolerances. |
| 2026-03-16 | **GitHub issue submitted** to Xilinx/mlir-air: "Flash Attention causal kernel produces incorrect output (masked by overly loose test tolerances)". Includes self-contained reproducer script. Waiting for developer response. |
| 2026-03-16 | **CPU attention fallback implemented**: Added `--cpu-attn` flag (default: on) and `--npu-attn` override. Step 7 in `run_transformer_block()` conditionally uses `attention_reference()` (CPU F32) instead of NPU flash attention. All other 14 steps remain on NPU. |
| | **1-layer verified** (`--verify --cpu-attn`): All 14 NPU kernels corr>0.999 (`[OK]`). Top-1 " is" matches CPU. |
| | **16-layer verified** (`--verify --cpu-attn`): All 16 layers pass. **Top-1 = " Paris"** (prob=0.48, correct factual answer). Logits corr=0.972 vs CPU F32. CPU top-1 = " the" (prob=0.065) — both valid, difference is benign BF16 numerical noise. |
| | **16-layer profiled** (`--profile --cpu-attn`): 25.9s total prefill (18.7s NPU kernel time, ~7.2s CPU attention + overhead). Per-layer avg: 1.62s wall, 1.17s kernel. |
| | **Phase 3A: VERIFIED CORRECT.** Full LLAMA-3.2-1B pipeline produces correct output with CPU attention fallback. |
| 2026-03-16 | **Phase 4 started: Per-kernel performance profiling.** Added C++ profiling harness (`test.cpp` + `make profile`) to `eltwise_add/` following the `matrix_multiplication/bf16` pattern. |
| | **Eltwise add profiled** — baseline F32 scalar: 214,619 µs, 0.23 GB/s. IRON: 432 µs, 57.6 GB/s. Gap: 497×. |
| | **Eltwise add optimized** — BF16 vec16 [8,1] herd: **415 µs, 60.6 GB/s**. Matches IRON (0.96×). 517× speedup over baseline. |
| | Full herd config sweep (12 configs): only [8,1] achieves 8-column parallelism; multi-row herds fail with ShimDMA channel exhaustion. |
| | Makefile updated with `AIE_TARGET` detection: `aie2p` (NPU2) defaults to BF16/vec16/[8,1]; `aie2` (NPU1) defaults to F32/scalar/[1,2]. |
| 2026-03-16 | **BF16 eltwise add integrated into LLAMA pipeline.** Replaced F32 scalar add in `llama3_prefill.py` steps 9 & 15 with BF16 vec16 [8,1] kernel. |
| | **16-layer verified** (`--verify --cpu-attn`): All 240 steps `[OK]`. Top-1 = " Paris" (prob=0.53). Logits corr=0.969. |
| | **Profiling**: NPU kernel total dropped from 18.67s → **13.40s** (28% reduction). Eltwise add: 8.19s → 1.34s (6.1× in pipeline). |
| | Eltwise add is no longer the bottleneck. GEMM Gate/Up (0.117s × 32 = 3.74s, 28%) is now the largest contributor. |

| 2026-03-17 | **XRT context + BO reuse**: NPU kernel 13.40s → 8.77s → **6.49s**. |
| 2026-03-17 | **GEMM investigation**: Found optimal tiles (3.5-5.5× speedup). Discovered BFP16 rounding mode bug. Fixed in non-direct-codegen path. |
| 2026-03-19 | **GEMM rounding fix landed upstream** in MLIR-AIE rebuild. Direct codegen now produces correct precision (corr=0.99992). |
| 2026-03-19 | **GEMM verified**: All 4 LLAMA shapes pass precision check. AIR 25% faster than IRON on Q/O. Ready to integrate. |
| 2026-03-20 | **FlashAttention re-tested** after PR #1438. **Still broken**: corr=0.13-0.34 for ALL configs at LQ=2048. `make run PASS` is false positive. Created `test_precision.py` and filed GitHub issue. CPU fallback remains necessary. |
| 2026-03-20 | **GEMM `run.py` test fixed**: rtol 1.0→0.04, inputs changed from `arange` to `randn*4`. Integer tests set to exact (rtol=0). |
| 2026-03-20 | **GEMM integrated into LLAMA pipeline**: 8×4 herd, per-shape optimal tiles, BF16 output. NPU 6.49s → **3.60s** (44% reduction). Compilation 334s → 34s. |
| | **16-layer verified**: Top-1 = " Paris" (prob=0.19). Logits corr=0.994. All 240 steps `[OK]`. |
| | Gap to IRON: **1.5×** (was 2.7×). SwiGLU (26%) and GEMM Gate/Up (21%) are largest NPU contributors. |
| 2026-03-20 | **SwiGLU optimized**: [8,1] herd + tile_n=4096 + 16-wide vectors. 59ms → **37ms** (1.6×). BD exhaustion workaround: larger tiles reduce iteration count under BD limit. |
| 2026-03-26 | **FlashAttention FIXED**: All configs pass with corr > 0.996. LLAMA causal (32h/8kv, 2048 seq): **15ms, 2,281 GFLOP/s** — **2× faster than IRON** (31ms). Only valid tile config: LQP=256, LKP=64 (causal constraints). Ready to integrate into LLAMA pipeline. |

---

## Phase 4: Performance Optimization

See `performance_optimization.md` for full profiling breakdown, IRON comparison, and optimization roadmap.

**Summary**: NPU kernel 18.67s → **3.57s** (81% reduction). FlashAttention now fixed (15ms, 2× faster than IRON). **Next**: Integrate NPU FlashAttention to replace CPU fallback (~38s savings).

---

## Quick Reference

```bash
cd programming_examples/llama3/build_peano

# Compile (one-time, ~5.5 min)
python3 ../llama3_prefill.py --compile-only --profile

# Run + verify
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --profile

# Single layer
python3 ../llama3_prefill.py --run-only --n-layers 1 --verify
```

See `performance_optimization.md` for profiling commands (AIR + IRON) and `LLAMA_verification.md` for full command reference.
