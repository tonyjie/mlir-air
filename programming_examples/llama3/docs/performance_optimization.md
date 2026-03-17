# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P). Reference implementation: IRON (`/home/jiajli/apps/IRON`).

### Current Status

| Metric | Baseline | Current | IRON | Gap to IRON |
|--------|----------|---------|------|-------------|
| NPU kernel total | 18.67s | **6.49s** | 2.32s | 2.8× |
| Wall time | 25.9s | ~47s | 2.84s | 16.5× |
| Per-layer (NPU) | 1.17s | 0.41s | 0.15s | 2.7× |

### Where the Time Goes (6.49s NPU breakdown)

| Category | Per-layer | ×16 layers | % of NPU | IRON (model) | Gap |
|----------|----------|------------|----------|-------------|-----|
| **FFN block** (Gate+Up+SwiGLU+Down) | 303ms | 4.85s | **75%** | 57.7ms | 5.3× |
| **GEMM Q/O** (×2) | 46ms | 0.74s | 11% | ~17ms | 2.7× |
| **Small kernels** (RMSNorm, RoPE, Add) | 33ms | 0.53s | 8% | ~14ms | 2.4× |
| **GEMM K/V** (×2) | 10ms | 0.16s | 2% | ~6ms | 1.7× |
| **CPU Attention** | 2,370ms | 37.9s | (wall only) | 42.7ms | 56× |

### Next Steps (ordered by impact)

| Priority | Action | Savings | Complexity | Status |
|----------|--------|---------|-----------|--------|
| **1** | Fix NPU flash attention | **37s** wall | Blocked upstream | Waiting |
| **2** | Optimize FFN block | **~4s** NPU | High (fusion or per-kernel) | Not started |
| **3** | Switch GEMM to BF16 output | **~0.5-1s** NPU | Low (one-arg change) | Ready |
| **4** | Vectorize RoPE / RMSNorm | **~0.3s** NPU | Medium (eltwise_add pattern) | Ready |
| **5** | GEMM tile/herd tuning | **~0.3s** NPU | Medium | Ready |

---

## AIR vs IRON — Kernel Breakdown Comparison

All numbers measured on the same NPU2, seq_len=2048. AIR: 2026-03-17, IRON: 2026-03-16.

### Understanding the Measurements

Each kernel invocation involves these phases:

```
bo.write()          — memcpy: numpy → BO (host pinned memory)
bo.sync(TO_DEVICE)  — DMA: host → NPU
kernel() + wait()   — actual NPU compute
bo.sync(FROM_DEV)   — DMA: NPU → host
bo.read()           — memcpy: BO → numpy
```

The two frameworks measure differently:

| Measurement | AIR (our `load_and_run`) | IRON standalone (pytest) | IRON model (`_execute_aie`) |
|-------------|-------------------------|------------------------|----------------------------|
| BO allocation | No (cached) | No (pre-allocated) | No (pre-allocated) |
| `bo.write` / `write_buffer` | **Yes** | No | **Yes** |
| `bo.sync(TO_DEVICE)` | **Yes** | No (before timer) | No (before timer in `run_runlist`) |
| kernel + wait | **Yes** | **Yes** | **Yes** |
| `bo.sync(FROM_DEVICE)` | **Yes** | No (after timer) | No (after timer in `run_runlist`) |
| `bo.read` / `read_buffer` | **Yes** | No | **Yes** (in `_execute_aie`) |

**For apple-to-apple comparison**, we use IRON's model-level timing (`_execute_aie_operation`) which includes write + kernel + read, comparable to our `load_and_run`.

Phase breakdown example (eltwise add, 4M BF16 elements, with BO reuse):

| Phase | AIR (µs) | Notes |
|-------|---------|-------|
| `bo.write()` | 1,677 | memcpy 3 × 8MB numpy → BO |
| `bo.sync(TO_DEVICE)` | 211 | DMA to NPU |
| **kernel + wait** | **482** | Actual NPU compute |
| `bo.sync(FROM_DEVICE)` | 226 | DMA from NPU |
| `bo.read()` | 2,901 | memcpy 3 × 8MB BO → numpy |
| **Total** | **5,498** | |

Kernel compute is only **9%** of per-invocation time. Host-side memory copies (`bo.write` + `bo.read`) dominate at 83%.

### Per-Invocation Comparison — Apple-to-Apple (including data transfer)

IRON model numbers from `_execute_aie_operation` (includes write_buffer + run_runlist + read_buffer).
AIR numbers from `load_and_run` profiler (includes bo.write + sync + kernel + sync + read).

Source: `docs/iron_profile_20260316.csv`, `docs/profiling_results_20260317.txt`

| # | Kernel | Shape | AIR (ms) | IRON model (ms) | Gap | IRON standalone (ms) |
|---|--------|-------|---------|-----------------|-----|---------------------|
| 1 | RMSNorm | (2048, 2048) + weight | 10 | 4.3 | 2.3× | 0.88 |
| 2 | GEMM Q | (2048, 2048) × (2048, 2048) → F32 | 23 | 8.8* | 2.6× | 7.4 |
| 3 | GEMM K | (2048, 2048) × (2048, 512) → F32 | 5 | 8.8* | 0.6× | 1.65 |
| 4 | GEMM V | (2048, 2048) × (2048, 512) → F32 | 5 | 8.8* | 0.6× | 1.65 |
| 5 | RoPE Q | (65536, 64) BF16 | 11 | 5.0 | 2.2× | 0.74 |
| 6 | RoPE K | (16384, 64) BF16 | 3 | 5.0 | 0.6× | 0.23 |
| 7 | Attention | Q(32,2048,64) K(8,2048,64) V(8,2048,64) | **2,370** (CPU) | **42.7** (NPU) | 56× | 36.8 |
| 8 | GEMM O | (2048, 2048) × (2048, 2048) → F32 | 23 | 8.8* | 2.6× | 7.4 |
| 9 | Add | (4194304,) BF16 | 6 | 4.2 | 1.4× | 0.43 |
| 10 | RMSNorm | (2048, 2048) + weight | 10 | 4.3 | 2.3× | 0.88 |
| 11 | GEMM Gate | (2048, 2048) × (2048, 8192) → F32 | 86 | — | — | — |
| 12 | GEMM Up | (2048, 2048) × (2048, 8192) → F32 | 86 | — | — | — |
| 13 | SwiGLU | SiLU(gate)×up, n=16M, BF16 | 58 | — | — | — |
| 14 | GEMM Down | (2048, 8192) × (8192, 2048) → F32 | 72 | — | — | — |
| 15 | Add | (4194304,) BF16 | 6 | 4.2 | 1.4× | 0.43 |
| | **FFN block (steps 10-15)** | | **318** | **57.7** | **5.5×** | — |
| | **Layer total (NPU only, no attn)** | | **405** | — | — | — |

\*IRON GEMM `_execute_aie_operation` avg = 8.75ms across 65 calls (mix of Q/K/V/O shapes). Per-shape breakdown not available from model profiling.

### 16-Layer Totals

| Metric | AIR | IRON | Gap |
|--------|-----|------|-----|
| **Total prefill (wall)** | ~47s | **2.84s** | 16.5× |
| **NPU kernel total** | **6.49s** | 2.32s (`run_runlist`) | 2.8× |
| **Per-layer (NPU kernel)** | 0.41s | 0.15s | 2.7× |
| All operator `_execute_aie` total | — | 2.75s (`__call__`) | — |
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

| Kernel | Shape | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | IRON model | IRON standalone |
|--------|-------|----------|-----------|------------|-----------|------------|-----------------|
| GEMM Gate/Up | 2048×2048×8192 | 110 | 117 | 111 | **86** | (fused) | — |
| GEMM Down | 2048×8192×2048 | 103 | 102 | 91 | **72** | (fused) | — |
| SwiGLU | n=16M, BF16 | 86 | 89 | 77 | **58** | (fused) | — |
| GEMM Q/O | 2048×2048×2048 | 51 | 54 | 33 | **23** | 8.8* | 7.4 |
| RoPE Q | 65536×64 | 27 | 56 | 16 | **11** | 5.0 | 0.74 |
| RMSNorm | (2048, 2048) | 28 | 30 | 15 | **10** | 4.3 | 0.88 |
| Eltwise Add | n=4M | **256** (F32) | **42** (BF16) | 13 | **6** | 4.2 | 0.43 |
| GEMM K/V | 2048×2048×512 | 21 | 40 | 8 | **5** | 8.8* | 1.65 |
| RoPE K | 16384×64 | 16 | 21 | 4 | **3** | 5.0 | 0.23 |

\*IRON GEMM model avg = 8.75ms across 65 calls (mix of Q/K/V/O shapes).

### Totals Progression

| Metric | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | IRON |
|--------|----------|-----------|------------|-----------|------|
| NPU kernel total | 18.67s | 13.40s | 8.77s | **6.49s** | 2.32s |
| NPU per-layer | 1.17s | 0.84s | 0.55s | **0.41s** | 0.15s |
| Wall time | 25.9s | 51.4s | ~47s | ~47s | 2.84s |
| Gap to IRON (NPU) | 8.1× | 5.8× | 3.8× | **2.8×** | 1.0× |
| Top-1 prediction | " Paris" | " Paris" | " Paris" | " Paris" | — |

Note: Wall time unchanged after BO reuse because CPU attention (~38s) dominates. NPU kernel time improved 65% from baseline.

---

## Remaining Per-Kernel Optimization Targets

**Overall target**: NPU 6.49s → ~2.3s = **4.2s savings** (to match IRON)

Using apple-to-apple comparison (both including data transfer):

| Priority | Kernel | Shape | AIR (ms) | IRON model (ms) | Gap | Action |
|----------|--------|-------|---------|-----------------|-----|--------|
| **High** | FFN block (steps 11-14) | 2048×2048×8192 (fused) | 302 | 57.7 | 5.2× | Kernel fusion |
| **Medium** | GEMM Q/O | 2048×2048×2048 | 23 | 8.8 | 2.6× | BF16 output, tile/herd tuning |
| **Medium** | RoPE Q | (65536, 64) BF16 | 11 | 5.0 | 2.2× | Vectorize, [8,1] herd |
| **Medium** | RMSNorm | (2048, 2048) BF16 | 10 | 4.3 | 2.3× | Vectorize, [8,1] herd |
| **Low** | Eltwise Add | (4194304,) BF16 | 6 | 4.2 | 1.4× | Nearly matched |
| **Low** | GEMM K/V | 2048×2048×512 | 5 | 8.8 | 0.6× | Already faster than IRON |
| **Low** | RoPE K | (16384, 64) BF16 | 3 | 5.0 | 0.6× | Already faster than IRON |
| **Blocked** | Attention (MHA) | Q(32,2048,64) | 2,370 (CPU) | 42.7 | 56× | Upstream NPU kernel fix |
| **Low** | RoPE K | (16384, 64) BF16 | 4 | 0.23 | 17× | Vectorize |
| **Blocked** | Attention (MHA) | Q(32,2048,64) | 2,370 (CPU) | 42.7 | 56× | Upstream NPU kernel fix |

Note: IRON fuses Gate GEMM + Up GEMM + SiLU + mul + Down GEMM into a single SwiGLU prefill dispatch. See `kernels/gemm.md` for GEMM precision and accumulator analysis.

---

## Completed Optimizations

### 1. Eltwise Add: F32 Scalar → BF16 Vectorized (2026-03-16)

517× standalone speedup, matches IRON. BF16 16-wide vectorized, [8,1] herd.

**PR**: https://github.com/Xilinx/mlir-air/pull/1431. See `docs/kernels/eltwise_add.md` for full analysis.

### 2. XRT Context Reuse (2026-03-16)

Cache `(backend, invoker)` per kernel in `KernelCache._loaded`. Load once per unique kernel, reuse invoker. NPU total: 13.40s → 8.77s (34% reduction). Eliminated ~40ms per-invocation device/xclbin/context setup overhead.

### 3. Buffer Object (BO) Reuse (2026-03-17)

Pre-allocate BOs on first invocation per kernel, reuse on subsequent calls. Also sync instruction BO only once. NPU total: 8.77s → 6.49s (26% reduction). Eliminated ~5ms per-invocation BO allocation overhead.

Combined XRT context + BO reuse: 13.40s → 6.49s (**52% total reduction**).

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
