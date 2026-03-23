# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P).

Reference: IRON (`/home/jiajli/apps/IRON`). IRON profiling data: `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`.

---

## End-to-End Prefill Comparison

| Metric | AIR (current) | IRON | Notes |
|--------|--------------|------|-------|
| **NPU kernel total** | **3.57s** | **~2.4s** | **Gap: 1.5×** |
| **Wall time** | **~44s** | **2.75s** | AIR uses CPU attention (~38s overhead) |
| Per-layer (NPU kernel) | 0.22s | 0.15s | 1.5× |
| Compilation | 37s | cached | One-time cost |
| Top-1 prediction | " Paris" ✓ | — | Correct factual answer |
| Logits corr vs CPU F32 | 0.994 | — | |

**Note on wall time**: AIR's 44s wall time is dominated by CPU attention fallback (~38s). IRON runs attention on NPU (MHA, 37ms/layer). Excluding attention:

| Metric | AIR (excl. attention) | IRON (excl. attention) | Gap |
|--------|----------------------|----------------------|-----|
| NPU kernels (non-attn) | **3.57s** | ~1.8s (total minus MHA) | **2.0×** |
| Per-layer (non-attn) | 0.22s | ~0.11s | 2.0× |

---

## Per-Kernel Comparison (per-invocation, in LLAMA pipeline)

AIR numbers include BO data transfer (write + sync + kernel + sync + read).
IRON model numbers from `operator.forward()` (includes write_buffer + run_runlist + read_buffer).

| Step | Kernel | Shape | AIR (ms) | IRON model (ms) | Gap |
|------|--------|-------|---------|-----------------|-----|
| 1,10 | RMSNorm | (2048, 2048) | 10 | 4.3 | 2.3× |
| 2 | GEMM Q | 2048×2048×2048 | 10 | 10.6* | 1.0× |
| 3 | GEMM K | 2048×2048×512 | 6 | 10.6* | — |
| 4 | GEMM V | 2048×2048×512 | 6 | 10.6* | — |
| 5 | RoPE Q | (65536, 64) | 17 | 5.3 | 3.2× |
| 6 | RoPE K | (16384, 64) | 4 | 5.3 | — |
| 7 | **Attention** | 32 heads, 2048 seq | **2,400 (CPU)** | **37.0 (NPU)** | **65×** |
| 8 | GEMM O | 2048×2048×2048 | 10 | 10.6* | 1.0× |
| 9,15 | Eltwise Add | (4,194,304) BF16 | 10 | 4.5 | 2.2× |
| 11 | GEMM Gate | 2048×2048×8192 | 24 | (fused) | — |
| 12 | GEMM Up | 2048×2048×8192 | 24 | (fused) | — |
| 13 | SwiGLU | SiLU(gate)×up, 16.7M | 37 | (fused) | — |
| 14 | GEMM Down | 2048×8192×2048 | 27 | (fused) | — |
| 11-14 | **FFN block** | | **112** | **57.4** (fused) | **2.0×** |

\*IRON GEMM model avg = 10.6ms across 65 calls (mix of Q/K/V/O shapes).

### Per-Kernel Standalone Comparison (kernel dispatch + wait only)

Standalone measurements isolate kernel execution from data transfer overhead.

| Kernel | AIR (µs) | IRON (µs) | AIR vs IRON |
|--------|---------|----------|-------------|
| GEMM Q/O (2048³) | **2,822** | 3,539 | **1.25× faster** |
| GEMM K/V (2048×2048×512) | **722** | 762 | **1.06× faster** |
| GEMM Gate/Up (2048×2048×8192) | **10,865** | 14,322 | **1.32× faster** |
| GEMM Down (2048×8192×2048) | **10,230** | 12,536 | **1.23× faster** |
| Eltwise Add (4M BF16) | **415** | 429 | **1.03× faster** |
| SwiGLU fused (5 ops) | — | 48,100 | — |
| MHA (2048 seq, 32 heads) | — (CPU) | 30,989 | — |
| RMSNorm (4M BF16) | — | 843 | — |
| RoPE Q (65536×64) | — | 845 | — |
| RoPE K (16384×64) | — | 257 | — |

AIR GEMM kernels are **23-32% faster** than IRON standalone. Eltwise add is matched.

---

## Blocked Items

| Item | Issue | Workaround |
|------|-------|-----------|
| **NPU Flash Attention** | Kernel produces uncorrelated output (corr=0.13-0.34) for ALL configs. `make run PASS` is false positive. GitHub issue filed. | CPU `attention_reference()` fallback (`--cpu-attn`) |

## Next Steps

| Priority | Action | Savings | Status |
|----------|--------|---------|--------|
| **1** | Fix NPU flash attention | **38s** wall | Blocked upstream |
| **2** | Vectorize RoPE / RMSNorm | ~0.3s NPU | Ready |
| **3** | FFN kernel fusion | ~1s NPU | Complex |

---

## Optimization History

### Totals Progression

| Metric | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | +GEMM/SwiGLU Opt | IRON |
|--------|----------|-----------|------------|-----------|-----------------|------|
| NPU kernel total | 18.67s | 13.40s | 8.77s | 6.49s | **3.57s** | ~2.4s |
| NPU per-layer | 1.17s | 0.84s | 0.55s | 0.41s | **0.22s** | 0.15s |
| Wall time | 25.9s | 51.4s | ~47s | ~47s | **~44s** | 2.75s |
| Gap to IRON (NPU) | 7.8× | 5.6× | 3.7× | 2.7× | **1.5×** | 1.0× |
| Compilation | 334s | 334s | 334s | 334s | **37s** | — |

NPU kernel time reduced **81%** from baseline (18.67s → 3.57s). Gap to IRON narrowed from 7.8× to **1.5×**.

### Per-Kernel Progression (per-invocation avg, ms)

| Kernel | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | Current | IRON model |
|--------|----------|-----------|------------|-----------|---------|------------|
| GEMM Gate/Up | 110 | 117 | 111 | 86 | **24** | (fused) |
| SwiGLU | 86 | 89 | 77 | 58 | **37** | (fused) |
| GEMM Down | 103 | 102 | 91 | 72 | **27** | (fused) |
| GEMM Q/O | 51 | 54 | 33 | 23 | **10** | 10.6 |
| RoPE Q | 27 | 56 | 16 | 11 | **17** | 5.3 |
| RMSNorm | 28 | 30 | 15 | 10 | **10** | 4.3 |
| Eltwise Add | **256** (F32) | **42** (BF16) | 13 | 6 | **10** | 4.5 |
| GEMM K/V | 21 | 40 | 8 | 5 | **6** | (in 10.6) |
| RoPE K | 16 | 21 | 4 | 3 | **4** | 5.3 |

---

## Completed Optimizations

### 1. Eltwise Add: F32 Scalar → BF16 Vectorized (2026-03-16)

517× standalone speedup, matches IRON. BF16 16-wide vectorized, [8,1] herd.
**PR**: https://github.com/Xilinx/mlir-air/pull/1431. See `kernels/eltwise_add.md`.

### 2. XRT Context Reuse (2026-03-16)

Cache `(backend, invoker)` per kernel. NPU total: 13.40s → 8.77s (34% reduction).

### 3. Buffer Object (BO) Reuse (2026-03-17)

Pre-allocate BOs per kernel. NPU total: 8.77s → 6.49s (26% reduction).

### 4. GEMM Optimization + Integration (2026-03-17 — 2026-03-20)

Per-shape optimal tiles (8×4 herd, BF16 output). BFP16 rounding mode fix landed upstream. NPU total: 6.49s → 3.60s (44% reduction). See `kernels/gemm.md`.

### 5. SwiGLU Optimization (2026-03-20)

[8,1] herd + tile_n=4096 + 16-wide vectors. 59ms → 37ms (1.6×). BD exhaustion workaround: larger tiles to reduce iteration count under BD limit. See `kernels/swiglu.md`.

---

## Known Limitations

### AIR DMA Buffer Descriptor (BD) Exhaustion

AIR's `dma_memcpy_nd` generates per-iteration BD chains that scale linearly with `n / tile_n`. The hardware has a fixed BD limit (~48 per MemTile). This prevents scaling to more AIE columns for large buffers:

| Buffer size | tile_n=2048, [8,1] iters | Compiles? |
|------------|-------------------------|-----------|
| 4.2M (eltwise_add) | 256 | **OK** |
| 15.7M | 960 | **OK** (max safe) |
| 16.8M (SwiGLU) | 1024 | **FAIL** |

**Workaround**: Increase tile_n to reduce iteration count. tile_n=4096 at 16.8M → 512 iterations → fits.

**IRON avoids this** via ObjectFIFO, which generates repeating BD patterns (fixed ~2 BDs per buffer regardless of iteration count).

---

## Profiling Commands

```bash
# AIR: Full 16-layer profiling
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile --cpu-attn

# AIR: Compile + run
python3 ../llama3_prefill.py --compile-only --cpu-attn
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --cpu-attn

# IRON: End-to-end model
cd /home/jiajli/apps/IRON && source ironenv/bin/activate
python iron/applications/llama_3.2_1b/inference.py \
    <weights> <tokenizer> --num_tokens 1 --prompt_len 2048 --profile -vv

# IRON: Per-kernel benchmarks
pytest iron/operators/ -m "llama and extensive" --build-dir build_llama -v -s
```

---

## Per-Kernel Analysis Docs

- `kernels/eltwise_add.md` — Herd sweep, IRON comparison, correctness verification
- `kernels/gemm.md` — Tile optimization, precision analysis, rounding mode investigation
- `kernels/swiglu.md` — [8,1] optimization, BD exhaustion analysis
- `kernels/flash_attention.md` — Correctness investigation, precision metrics, status
