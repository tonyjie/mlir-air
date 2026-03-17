# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

This document tracks performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P). The reference implementation is IRON (`/home/jiajli/apps/IRON`).

**Current status**: NPU kernel time = 13.40s, wall time = 51.4s. IRON reference = 2.84s (measured on our NPU2).

---

## IRON Reference Profiling (measured on our NPU2, 2026-03-16)

All numbers measured at 2048-token scale using:
```bash
cd /home/jiajli/apps/IRON
pytest iron/operators/<op>/test.py -m "llama and extensive" -v -s
```

### Per-Kernel Latencies

| Kernel | Shape | IRON Latency (µs) | pytest test ID |
|--------|-------|-------------------|----------------|
| GEMM Q/O | 2048×2048×2048, 8-col | 7,367 | `gemm_2048x2048x2048_64x64x64_8cols_bcolmaj_ccolmaj` |
| GEMM K/V | 2048×2048×512, 8-col | 1,650 | `llama_kv_proj_2048tok` |
| SwiGLU Prefill (fused) | 2048×2048×8192 | 80,500 | `llama_swiglu_prefill_2048tok_2048x8192` |
| MHA | seq=2048, 32 heads | 36,750 | `llama_prefill_2048tok` |
| RMSNorm | 4,194,304 BF16 | 880 | `llama_prefill_rms_norm_2048tok` |
| RoPE Q | 65536×64 BF16 | 738 | `llama_prefill_q_2048tok` |
| Eltwise Add | 4,194,304 BF16 | 432 | `llama_prefill_add_2048tok` |
| RoPE K | 16384×64 BF16 | 233 | `llama_prefill_k_2048tok` |

### IRON Per-Layer Estimate (from isolated kernel sum)

| Block | Components | Total |
|-------|-----------|-------|
| Attention | RMSNorm (0.88ms) + GEMM Q (7.4ms) + GEMM K (1.7ms) + GEMM V (1.7ms) + RoPE Q (0.74ms) + RoPE K (0.23ms) + MHA (36.8ms) + GEMM O (7.4ms) + Add (0.43ms) | **57.3ms** |
| FFN | RMSNorm (0.88ms) + SwiGLU fused (80.5ms) + Add (0.43ms) | **81.8ms** |
| **Total per layer** | | **139.1ms** |
| **16 layers (kernel only)** | | **2.2s** |
| **16 layers (measured)** | | **2.91s** (0.7s Python/XRT overhead) |

### IRON End-to-End Model Profiling (measured on our NPU2, 2026-03-16)

Run command:
```bash
cd /home/jiajli/apps/IRON/iron/applications/llama_3.2_1b
PYTHONPATH=/home/jiajli/apps/IRON:$PYTHONPATH python3 inference.py \
    <weights> <tokenizer> --num_tokens 1 --prompt_len 2048 --profile -vv
```

Analyze with: `python3 analyze_profile.py logs/profile_<timestamp>.log`

| Function | Calls | Total (s) | Avg (ms) | Notes |
|----------|-------|-----------|----------|-------|
| `generate` | 1 | **2.842** | — | Total prefill |
| `model.forward` | 1 | 2.830 | — | |
| `run_runlist` (NPU dispatch) | 194 | **2.324** | 12.0 | Pure NPU kernel time |
| `aie_base.__call__` | 194 | 2.745 | 14.2 | NPU + Python overhead |
| `transformer.forward` | 16 | 3.122 | 195.1 | Per-block (includes sub-calls) |
| `feed_forward.forward` | 16 | 2.459 | 153.7 | FFN block |
| `gqa.forward` | 16 | 1.331 | 83.2 | Attention block |
| `swiglu_prefill._execute_aie` | 16 | 0.923 | **57.7** | Fused gate+up+act+down |
| `gemm.forward` | 65 | 0.685 | 10.5 | Q/K/V/O projections |
| `mha._execute_aie` | 16 | 0.683 | **42.7** | Multi-head attention |
| `gemm._execute_aie` | 65 | 0.569 | 8.8 | Pure GEMM dispatch |

**Breakdown**: 2.84s total = 2.32s NPU dispatch + 0.52s Python overhead (18%).

### IRON Architecture Notes

- **Persistent XRT context**: Device, xclbin, kernel handles created once via `AIEContext.prepare_runtime()`, reused across all dispatches. No per-invocation load/unload. 194 dispatches share one context.
- **SwiGLU fusion**: Gate GEMM + Up GEMM + SiLU activation + element-wise multiply + Down GEMM fused into single kernel dispatch (57.7ms total vs our 4 separate dispatches at 308ms).
- **ObjectFIFO-based DMA**: Compiler-managed DMA pipelining with double-buffering. Our AIR dialect uses explicit `dma_memcpy_nd` without overlap.
- **C++ kernels with compiler hints**: `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_MIN_ITERATION_COUNT` for loop optimization.

---

## Our Current Profiling (2026-03-16)

### Per-Kernel Breakdown

```bash
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile --cpu-attn
```

| Kernel | Avg (ms) | Count | Total (s) | % of NPU | IRON (ms) | Overhead est. |
|--------|----------|-------|-----------|----------|-----------|---------------|
| GEMM Gate/Up | 117 | x32 | 3.74 | **28%** | (fused) | ~40ms XRT + ~77ms kernel |
| GEMM Q/O | 54 | x32 | 1.73 | 13% | 7.4 | ~40ms XRT + ~14ms kernel |
| GEMM Down | 102 | x16 | 1.63 | 12% | (fused) | ~40ms XRT + ~62ms kernel |
| SwiGLU | 89 | x16 | 1.42 | 11% | (fused) | ~40ms XRT + ~49ms kernel |
| Eltwise Add | 42 | x32 | 1.34 | 10% | 0.43 | ~40ms XRT + ~2ms kernel |
| GEMM K/V | 40 | x32 | 1.28 | 10% | 1.65 | ~40ms XRT + ~0ms kernel |
| RMSNorm | 30 | x33 | 0.99 | 7% | 0.88 | ~40ms XRT + ~0ms kernel |
| RoPE Q | 56 | x16 | 0.90 | 7% | 0.74 | ~40ms XRT + ~16ms kernel |
| RoPE K | 21 | x16 | 0.34 | 3% | 0.23 | ~40ms XRT + ~0ms kernel |
| **NPU Total** | | **225** | **13.40** | | | |
| CPU Attention | 2,370 | x16 | 37.95 | — | 36.8 | CPU fallback |

### Summary

| Metric | Ours | IRON | Gap |
|--------|------|------|-----|
| Total prefill (wall) | 51.4s | **2.84s** | 18.1× |
| NPU kernel total | 13.40s | 2.32s (run_runlist) | 5.8× |
| XRT overhead (estimated) | ~9.0s | 0.52s | 17.3× |
| Kernel compute (estimated) | ~4.4s | 2.32s | 1.9× |
| Per-layer (wall) | 3.21s | 0.18s | 17.8× |
| Per-layer (NPU kernel) | 0.84s | 0.15s | 5.6× |

---

## Optimization Roadmap

### Priority 1: Fix NPU Flash Attention (upstream bug)

**Impact**: 37.9s → 0.59s = **37.3s savings** (73% of wall time)

CPU attention (`attention_reference()`) is used as fallback because the NPU flash attention kernel has a correctness bug (corr=0.31 vs standard attention). GitHub issue filed upstream.

- IRON MHA at 2048 tokens: 36.8ms per invocation (NPU)
- Our CPU fallback: 2,370ms per invocation
- **64× slower** than NPU

**Status**: Blocked on upstream fix. See `LLAMA_flash_attention.md`.

### Priority 2: XRT Context Reuse

**Impact**: ~9s → ~0s = **9s savings** (67% of NPU time)

Each of 225 NPU kernel invocations pays ~40ms for XRT device init + xclbin register + hw_context creation + kernel handle + instruction BO allocation. IRON avoids this by creating a persistent `AIEContext` that maintains the XRT device and kernel handles across all dispatches.

**Approach**:
- Cache loaded backends per kernel name in `KernelCache`
- Only load once per unique kernel; reuse invoker for subsequent calls
- Unload all at program end

**Expected result**: NPU time from 13.4s → ~4.4s (kernel compute only)

### Priority 3: Per-Kernel Compute Optimization

**Impact**: ~4.4s → ~2.2s = **2.2s savings** (estimated)

After removing XRT overhead, the remaining gap is actual kernel compute efficiency:

| Kernel | Ours (est. kernel only) | IRON | Gap | Optimization |
|--------|------------------------|------|-----|--------------|
| GEMM Gate/Up (×2) | ~77ms each | (fused) | — | Kernel fusion or larger herd |
| GEMM Down | ~62ms | (fused) | — | Kernel fusion |
| SwiGLU | ~49ms | (fused) | — | Kernel fusion |
| **FFN block total** | ~265ms | 80.5ms | **3.3×** | Fuse gate+up+act+down |
| GEMM Q/O | ~14ms | 7.4ms | 1.9× | Tile tuning, larger herd |
| RoPE Q | ~16ms | 0.74ms | 22× | Vectorize, more columns |
| Eltwise Add | ~2ms | 0.43ms | 4.7× | Already vectorized, XRT-limited |
| GEMM K/V | ~0ms | 1.65ms | — | Already fast (overhead-dominated) |
| RMSNorm | ~0ms | 0.88ms | — | Already fast (overhead-dominated) |
| RoPE K | ~0ms | 0.23ms | — | Already fast (overhead-dominated) |

Key compute gaps:
1. **FFN block (3.3×)**: IRON fuses 5 operations into one dispatch. Our 4 separate dispatches lose inter-op data locality.
2. **RoPE Q (22×)**: Our RoPE kernel likely needs vectorization and more columns (currently scalar or limited parallelism).
3. **GEMM Q/O (1.9×)**: Tile configuration and herd shape tuning.

---

## Completed Optimizations

### Eltwise Add: F32 Scalar → BF16 Vectorized (2026-03-16)

**Result**: 517× standalone speedup, matches IRON.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Standalone (C++ harness) | 214,619 µs | 415 µs | **517×** |
| In LLAMA pipeline | 256ms (8.19s total) | 42ms (1.34s total) | **6.1×** |
| % of NPU time | 44% | 10% | No longer bottleneck |
| vs IRON (standalone) | 497× slower | **0.96×** (4% faster) | Matched |

**Changes**:
- `eltwise_add.py`: Added BF16 vectorized compute path (16-wide `vector.transfer_read/write`), 2D herd support (`--herd-x 8 --herd-y 1`)
- `test.cpp` + `Makefile`: C++ profiling harness with `make profile`
- `AIE_TARGET` detection in Makefile for NPU1/NPU2 defaults

**PR**: https://github.com/Xilinx/mlir-air/pull/1431

See `docs/kernels/eltwise_add.md` for full analysis including herd config sweep.

---

## Profiling Commands

### Our pipeline
```bash
cd programming_examples/llama3/build_peano

# Full 16-layer profiling
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile --cpu-attn

# Single layer with verification
python3 ../llama3_prefill.py --run-only --n-layers 1 --verify --profile
```

### IRON per-kernel benchmarks (2048-token scale)
```bash
cd /home/jiajli/apps/IRON

# All kernels at 2048 tokens
pytest iron/operators/ -m "llama and extensive" -v -s

# Individual kernels
pytest iron/operators/gemm/test.py -k "gemm_2048x2048x2048" -v -s
pytest iron/operators/gemm/test.py -k "llama_kv_proj_2048tok" -v -s
pytest iron/operators/rms_norm/test.py -m "llama and extensive" -v -s
pytest iron/operators/rope/test.py -m "llama and extensive" -v -s
pytest iron/operators/elementwise_add/test.py -m "llama and extensive" -v -s
pytest iron/operators/mha/test.py -m "llama and extensive" -v -s
pytest iron/operators/swiglu_prefill/test.py -m "llama and extensive" -v -s
```

### Standalone kernel profiling (C++ harness)
```bash
cd programming_examples/eltwise_add
make profile                    # NPU2 BF16 vec16 [8,1], N=4M
make profile AIE_TARGET=aie2    # NPU1 F32 scalar [1,2], N=64K
```

---

## Per-Kernel Analysis Docs

Detailed per-kernel optimization analysis is in `docs/kernels/`:
- `kernels/eltwise_add.md` — Herd sweep, IRON comparison, correctness verification
- (Future: `kernels/gemm.md`, `kernels/rope.md`, etc.)
