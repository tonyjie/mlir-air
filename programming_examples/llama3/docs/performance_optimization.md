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
| **2** | **Integrate GEMM optimal tiles + 8×4 herd** | **~4s** NPU | **Low** | **Ready** |
| **3** | Vectorize RoPE / RMSNorm | **~0.3s** NPU | Medium (eltwise_add pattern) | Ready |
| **4** | FFN kernel fusion | **~2s** NPU | High | Not started |

### Blocked Items

| Item | Issue | Workaround |
|------|-------|-----------|
| **NPU Flash Attention** | Kernel still produces uncorrelated output (corr=0.13-0.34 vs standard attention for ALL configs at LQ=2048). `make run PASS` is a false positive — element-wise tolerance can't detect wrong attention patterns. GitHub issue filed. | CPU `attention_reference()` fallback (`--cpu-attn`) |

### Actionable Now

| Item | Action | Impact |
|------|--------|--------|
| **GEMM integration** | Update `_build_gemm_module()` with 8×4 herd + per-shape tiles | **~4s NPU savings** (3.5-5.5× faster GEMM) |
| **RoPE vectorization** | Apply eltwise_add pattern (BF16 vec16, [8,1] herd) | ~0.3s NPU savings |
| **RMSNorm vectorization** | Same pattern | ~0.3s NPU savings |

Note: GEMM rounding mode fix landed in upstream MLIR-AIE rebuild (2026-03-19). Direct codegen now produces correct precision (corr=0.99992, 8% fail at 4% tolerance). See `kernels/gemm.md` for full analysis.

---

## AIR vs IRON — Kernel Breakdown Comparison

AIR: measured 2026-03-19. IRON: from `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md` (prio=False, emu=True, correct LLAMA config).

### Per-Kernel Standalone Comparison (kernel dispatch + wait only)

IRON standalone from `pytest --build-dir build_llama -m "llama and extensive"`.
AIR standalone from `test.exe` (best tile configs, 8×4 herd, direct codegen).

| Kernel | Shape | AIR (µs) | AIR GFLOP/s | IRON (µs) | IRON GFLOP/s | AIR corr | IRON corr | AIR vs IRON |
|--------|-------|---------|-------------|----------|-------------|---------|-----------|-------------|
| GEMM Q/O | 2048³ | **2,822** | 6,088 | 3,539 | 4,854 | 0.99992 | 0.99994 | **1.25× faster** |
| GEMM K/V | 2048×2048×512 | **722** | 5,949 | 762 | 5,634 | 0.99992 | 0.99994 | **1.06× faster** |
| SwiGLU Prefill | 2048×2048×8192 (fused) | — | — | 48,100 | — | — | 0.99972 | — |
| MHA | seq=2048, 32 heads | — (CPU) | — | 30,989 | — | — | 0.99758 | — |
| RMSNorm | 4M BF16 | — | — | 843 | — | — | 0.99998 | — |
| RoPE Q | 65536×64 | — | — | 845 | — | — | 0.99999 | — |
| RoPE K | 16384×64 | — | — | 257 | — | — | 0.99999 | — |
| Eltwise Add | 4M BF16 | 415 | 60,600 | 429 | 58,600 | 0.99998 | 0.99998 | **1.03× faster** |

### IRON Model-Level Timing (includes data transfer)

From IRON end-to-end run (2.75s prefill):

| Function | Calls | Avg/call (ms) |
|----------|-------|---------------|
| `transformer.forward` | 16 | 152.3 |
| `gqa.forward` | 16 | 76.9 |
| `swiglu_prefill.forward` | 16 | 57.4 |
| `gemm.forward` | 65 | 10.6 |
| `mha.forward` | 16 | 37.0 |
| `rope.forward` | 32 | 5.3 |
| `elementwise_add.forward` | 32 | 4.5 |
| `rms_norm.forward` | 33 | 4.3 |

### 16-Layer Totals

| Metric | AIR | IRON | Gap |
|--------|-----|------|-----|
| **Total prefill (wall)** | ~47s | **2.75s** | 17× |
| **NPU kernel total** | **6.49s** | ~2.4s (from model profile) | 2.7× |
| **Per-layer** | 0.41s | 0.15s | 2.7× |
| CPU attention total | ~37.9s | — | — |
| IRON NPU attention total | — | 0.59s (MHA) | — |

### IRON Architecture Advantages

- **Persistent XRT context**: All dispatches share one context (no per-call load/unload)
- **SwiGLU fusion**: 5 ops in one dispatch (57.4ms vs our ~300ms for 4 separate ops)
- **ObjectFIFO DMA**: Compiler-managed double-buffering (vs our explicit `dma_memcpy_nd`)
- **C++ kernel hints**: `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_MIN_ITERATION_COUNT`

### IRON Profiling Reference

Full IRON profiling data: `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`

```bash
# End-to-end model
cd /home/jiajli/apps/IRON
source ironenv/bin/activate
python iron/applications/llama_3.2_1b/inference.py \
    <weights> <tokenizer> --num_tokens 1 --prompt_len 2048 --profile -vv

# Per-kernel (use separate build dir to avoid cache conflicts)
pytest iron/operators/ -m "llama and extensive" --build-dir build_llama -v -s
```

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
| NPU kernel total | 18.67s | 13.40s | 8.77s | **6.49s** | ~2.4s |
| NPU per-layer | 1.17s | 0.84s | 0.55s | **0.41s** | 0.15s |
| Wall time | 25.9s | 51.4s | ~47s | ~47s | 2.75s |
| Gap to IRON (NPU) | 7.8× | 5.6× | 3.7× | **2.7×** | 1.0× |
| Top-1 prediction | " Paris" | " Paris" | " Paris" | " Paris" | — |

Note: Wall time unchanged after BO reuse because CPU attention (~38s) dominates. NPU kernel time improved 65% from baseline.

---

## Remaining Per-Kernel Optimization Targets

**Overall target**: NPU 6.49s → ~2.4s = **4.1s savings** (to match IRON)

| Priority | Kernel | AIR (ms) | IRON standalone (ms) | IRON model (ms) | Action | Status |
|----------|--------|---------|---------------------|-----------------|--------|--------|
| **Blocked** | Attention (MHA) | 2,370 (CPU) | 31.0 | 37.0 | Upstream kernel fix | Waiting |
| **Ready** | GEMM Q/O | 23 | 3.5 | 10.6 | Integrate 8×4 + optimal tiles | **Ready** |
| **Ready** | RoPE Q | 11 | 0.85 | 5.3 | Vectorize, [8,1] herd | Actionable |
| **Ready** | RMSNorm | 10 | 0.84 | 4.3 | Vectorize, [8,1] herd | Actionable |
| **Done** | Eltwise Add | 6 | 0.43 | 4.5 | Nearly matched | Complete |
| **Done** | GEMM K/V | 5 | 0.76 | (in 10.6 avg) | Already faster | — |
| **Future** | FFN fusion | 302 | 48.1 (fused) | 57.4 (fused) | Kernel fusion | Complex |

### GEMM Status

Rounding mode fix landed in upstream MLIR-AIE rebuild (2026-03-19). Direct codegen now produces correct precision.

- **Performance**: 8×4 herd + optimal tiles → Q/O: 2,822 µs (**25% faster** than IRON's 3,539 µs). K/V: 722 µs (6% faster than IRON's 762 µs).
- **Precision**: corr=0.99992, 4% fail=8.0% (IRON: 0.99994, 7.5%). Rounding bias eliminated (mean_signed ≈ 0).
- **Verification**: `run.py` tolerance fixed from rtol=1.0 to 0.04. Integer variants set to rtol=0 (exact).
- **Ready to integrate** into LLAMA pipeline.

See `kernels/gemm.md` for full analysis.

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

### 4. GEMM Investigation (2026-03-17)

**Performance**: Found optimal tile configs (8×4 herd, per-shape tiles) giving 3.5-5.5× standalone speedup. Q/O: 14,012 → 2,902 µs. Matches IRON at ~5,900 GFLOP/s.

**Precision**: Discovered BFP16 rounding mode bug — AIE2P default is `floor` rounding, causing systematic -0.065×K bias. Fixed in non-direct-codegen path by adding `set_rounding(conv_even)` to `mm_aie2p.cc`. After fix: corr=0.99992, 4% fail=8% (matching IRON's 7.4%).

**Verification**: Fixed `run.py` test tolerances (rtol 1.0→0.04 for BF16, 0 for integer). Old 100% tolerance was masking precision bugs.

See `kernels/gemm.md` for full analysis.

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
