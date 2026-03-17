# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P). Reference implementation: IRON (`/home/jiajli/apps/IRON`).

### Current Status

| Metric | Baseline | Current | IRON | Gap to IRON |
|--------|----------|---------|------|-------------|
| NPU kernel total | 18.67s | **8.77s** | 2.32s | 3.8× |
| Wall time | 25.9s | ~47s | 2.84s | 16.5× |
| Per-layer (NPU) | 1.17s | 0.55s | 0.15s | 3.7× |

### Where the Time Goes (8.77s NPU breakdown)

| Category | Per-layer | ×16 layers | % of NPU | IRON | Gap |
|----------|----------|------------|----------|------|-----|
| **FFN block** (Gate+Up+SwiGLU+Down) | 390ms | 6.24s | **71%** | 57.7ms | 6.8× |
| **GEMM Q/O** (×2) | 66ms | 1.06s | 12% | 14.8ms | 4.5× |
| **Small kernels** (RMSNorm, RoPE, Add) | 60ms | 0.96s | 11% | 3.1ms | 19× |
| **GEMM K/V** (×2) | 16ms | 0.26s | 3% | 3.3ms | 4.8× |
| **CPU Attention** | 2,370ms | 37.9s | (wall only) | 42.7ms | 56× |

### Next Steps (ordered by impact)

| Priority | Action | Savings | Complexity | Status |
|----------|--------|---------|-----------|--------|
| **1** | Fix NPU flash attention | **37s** wall | Blocked upstream | Waiting |
| **2** | Optimize FFN block | **~5.3s** NPU | High (fusion or per-kernel) | Not started |
| **3** | Switch GEMM to BF16 output | **~1-2s** NPU | Low (one-arg change) | Ready |
| **4** | Vectorize RoPE / RMSNorm | **~0.8s** NPU | Medium (eltwise_add pattern) | Ready |
| **5** | GEMM tile/herd tuning | **~0.5s** NPU | Medium | Ready |

---

## AIR vs IRON — Kernel Breakdown Comparison

All numbers measured on the same NPU2, seq_len=2048, 2026-03-16.

### Per-Invocation Comparison (with shapes)

| # | Kernel | Shape | AIR (ms) | IRON (ms) | Gap | Notes |
|---|--------|-------|---------|-----------|-----|-------|
| 1 | RMSNorm | (2048, 2048) + weight | 15 | 0.88 | 17× | |
| 2 | GEMM Q | (2048, 2048) × (2048, 2048) → F32 | 33 | 7.4 | 4.5× | |
| 3 | GEMM K | (2048, 2048) × (2048, 512) → F32 | 8 | 1.65 | 4.8× | |
| 4 | GEMM V | (2048, 2048) × (2048, 512) → F32 | 8 | 1.65 | 4.8× | |
| 5 | RoPE Q | (65536, 64) BF16 | 16 | 0.74 | 22× | |
| 6 | RoPE K | (16384, 64) BF16 | 4 | 0.23 | 17× | |
| 7 | Attention | Q(32,2048,64) K(8,2048,64) V(8,2048,64) | **2,370** (CPU) | **42.7** (NPU) | 56× | CPU fallback; IRON uses NPU MHA |
| 8 | GEMM O | (2048, 2048) × (2048, 2048) → F32 | 33 | 7.4 | 4.5× | |
| 9 | Add | (4194304,) BF16 | 13 | 0.43 | 30× | Kernel matched; BO alloc overhead |
| 10 | RMSNorm | (2048, 2048) + weight | 15 | 0.88 | 17× | |
| 11 | GEMM Gate | (2048, 2048) × (2048, 8192) → F32 | 111 | — | — | IRON fuses steps 11-14 |
| 12 | GEMM Up | (2048, 2048) × (2048, 8192) → F32 | 111 | — | — | into single SwiGLU prefill |
| 13 | SwiGLU | SiLU(gate)×up, n=16M, BF16 | 77 | — | — | dispatch: **57.7ms** total |
| 14 | GEMM Down | (2048, 8192) × (8192, 2048) → F32 | 91 | — | — | |
| 15 | Add | (4194304,) BF16 | 13 | 0.43 | 30× | |
| | **FFN block (steps 10-15)** | | **418** | **59.0** | **7.1×** | |
| | **Layer total (steps 1-15)** | | **2,918** | — | — | |
| | **Layer total (NPU only, no attn)** | | **548** | — | — | |

### 16-Layer Totals

| Metric | AIR | IRON | Gap |
|--------|-----|------|-----|
| **Total prefill (wall)** | ~47s | **2.84s** | 16.5× |
| **NPU kernel total** | **8.77s** | 2.32s | 3.8× |
| **Per-layer (NPU kernel)** | 0.55s | 0.15s | 3.7× |
| CPU attention total | ~37.9s | — | — |
| IRON NPU attention total | — | 0.68s | — |

### IRON Profiling Details

Per-kernel latencies measured with:
```bash
cd /home/jiajli/apps/IRON
pytest iron/operators/<op>/test.py -m "llama and extensive" -v -s
```

End-to-end model profiling:
```bash
cd /home/jiajli/apps/IRON/iron/applications/llama_3.2_1b
PYTHONPATH=/home/jiajli/apps/IRON:$PYTHONPATH python3 inference.py \
    <weights> <tokenizer> --num_tokens 1 --prompt_len 2048 --profile -vv
# Analyze: python3 analyze_profile.py logs/profile_<timestamp>.log
```

IRON end-to-end breakdown (2.84s total):

| Function | Calls | Total (s) | Avg (ms) |
|----------|-------|-----------|----------|
| `run_runlist` (pure NPU dispatch) | 194 | **2.32** | 12.0 |
| `swiglu_prefill._execute_aie` | 16 | 0.92 | 57.7 |
| `gemm._execute_aie` | 65 | 0.57 | 8.8 |
| `mha._execute_aie` | 16 | 0.68 | 42.7 |
| Python overhead (buffer sync, etc.) | — | 0.52 | — |

### IRON Architecture Advantages

- **Persistent XRT context**: 194 dispatches share one context (no per-call load/unload)
- **SwiGLU fusion**: 5 ops in one dispatch (57.7ms vs our 390ms for 4 separate ops)
- **ObjectFIFO DMA**: Compiler-managed double-buffering (vs our explicit `dma_memcpy_nd`)
- **C++ kernel hints**: `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_MIN_ITERATION_COUNT`

---

## Optimization History

### Per-Kernel Progression (per-invocation avg, ms)

| Kernel | Shape | Baseline | +BF16 Add | +XRT Reuse | IRON |
|--------|-------|----------|-----------|------------|------|
| GEMM Gate/Up | 2048×2048×8192 | 110 | 117 | **111** | (fused) |
| GEMM Down | 2048×8192×2048 | 103 | 102 | **91** | (fused) |
| SwiGLU | n=16M, BF16 | 86 | 89 | **77** | (fused) |
| GEMM Q/O | 2048×2048×2048 | 51 | 54 | **33** | 7.4 |
| RoPE Q | 65536×64 | 27 | 56 | **16** | 0.74 |
| GEMM K/V | 2048×2048×512 | 21 | 40 | **8** | 1.65 |
| RMSNorm | (2048, 2048) | 28 | 30 | **15** | 0.88 |
| Eltwise Add | n=4M | **256** (F32) | **42** (BF16) | **13** | 0.43 |
| RoPE K | 16384×64 | 16 | 21 | **4** | 0.23 |

### Totals Progression

| Metric | Baseline | +BF16 Add | +XRT Reuse | IRON |
|--------|----------|-----------|------------|------|
| NPU kernel total | 18.67s | 13.40s | **8.77s** | 2.32s |
| NPU per-layer | 1.17s | 0.84s | **0.55s** | 0.15s |
| Wall time | 25.9s | 51.4s | ~47s | 2.84s |
| Top-1 prediction | " Paris" | " Paris" | " Paris" | — |

Note: Wall time increased after BF16 add because the BF16 residual state changes CPU attention inputs, causing `attention_reference()` to take longer. NPU kernel time consistently improved.

---

## Remaining Per-Kernel Optimization Targets

**Overall target**: NPU 8.77s → ~2.3s = **6.5s savings** (to match IRON)

| Priority | Kernel | Shape | AIR (ms) | IRON (ms) | Gap | Action |
|----------|--------|-------|---------|-----------|-----|--------|
| **High** | FFN block (steps 11-14) | 2048×2048×8192 (fused) | 390 | 57.7 | 6.8× | Kernel fusion |
| **High** | RoPE Q | (65536, 64) BF16 | 16 | 0.74 | 22× | Vectorize, [8,1] herd |
| **Medium** | RMSNorm | (2048, 2048) BF16 | 15 | 0.88 | 17× | Vectorize, [8,1] herd |
| **Medium** | GEMM Q/O | 2048×2048×2048 | 33 | 7.4 | 4.5× | BF16 output, tile/herd tuning |
| **Medium** | GEMM K/V | 2048×2048×512 | 8 | 1.65 | 4.8× | BF16 output, tile/herd tuning |
| **Low** | Eltwise Add | (4194304,) BF16 | 13 | 0.43 | 30× | Kernel matched; BO alloc overhead |
| **Low** | RoPE K | (16384, 64) BF16 | 4 | 0.23 | 17× | Vectorize |
| **Blocked** | Attention (MHA) | Q(32,2048,64) | 2,370 (CPU) | 42.7 | 56× | Upstream NPU kernel fix |

Note: IRON fuses Gate GEMM + Up GEMM + SiLU + mul + Down GEMM into a single SwiGLU prefill dispatch. See `kernels/gemm.md` for GEMM precision and accumulator analysis.

---

## Completed Optimizations

### 1. Eltwise Add: F32 Scalar → BF16 Vectorized (2026-03-16)

517× standalone speedup, matches IRON. BF16 16-wide vectorized, [8,1] herd.

**PR**: https://github.com/Xilinx/mlir-air/pull/1431. See `docs/kernels/eltwise_add.md` for full analysis.

### 2. XRT Context Reuse (2026-03-16)

Cache `(backend, invoker)` per kernel in `KernelCache._loaded`. Load once, reuse invoker. NPU total: 13.40s → 8.77s (34% reduction). Biggest wins on small kernels (GEMM K/V: 40ms → 8ms, RoPE K: 21ms → 4ms).

---

## Profiling Commands

```bash
# AIR: Full 16-layer profiling
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile --cpu-attn

# AIR: Single layer with verification
python3 ../llama3_prefill.py --run-only --n-layers 1 --verify --profile

# IRON: Per-kernel benchmarks at 2048 tokens
cd /home/jiajli/apps/IRON
pytest iron/operators/gemm/test.py -k "gemm_2048x2048x2048" -v -s
pytest iron/operators/gemm/test.py -k "llama_kv_proj_2048tok" -v -s
pytest iron/operators/rms_norm/test.py -m "llama and extensive" -v -s
pytest iron/operators/rope/test.py -m "llama and extensive" -v -s
pytest iron/operators/elementwise_add/test.py -m "llama and extensive" -v -s
pytest iron/operators/mha/test.py -m "llama and extensive" -v -s
pytest iron/operators/swiglu_prefill/test.py -m "llama and extensive" -v -s

# IRON: End-to-end model
cd /home/jiajli/apps/IRON/iron/applications/llama_3.2_1b
PYTHONPATH=/home/jiajli/apps/IRON:$PYTHONPATH python3 inference.py \
    <weights> <tokenizer> --num_tokens 1 --prompt_len 2048 --profile -vv

# Standalone kernel profiling (C++ harness)
cd programming_examples/eltwise_add
make profile
```

---

## Per-Kernel Analysis Docs

Detailed per-kernel optimization analysis in `docs/kernels/`:
- `kernels/eltwise_add.md` — Herd sweep, IRON comparison, correctness verification
- (Future: `kernels/gemm.md`, `kernels/rope.md`, etc.)
