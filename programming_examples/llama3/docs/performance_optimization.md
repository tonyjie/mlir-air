# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P).

Reference: IRON (`/home/jiajli/apps/IRON`). IRON profiling data: `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`.

---

## End-to-End Prefill Comparison

| Metric | AIR (current) | IRON | Notes |
|--------|--------------|------|-------|
| **16 transformer layers** | **3.88s** | **2.44s** | **Gap: 1.59×** (same profiling scope — see below) |
| **Wall time** | **5.39s** | **2.75s** | AIR includes weight loading (~1.5s) + CPU LM Head |
| Per-layer avg | 243ms | 152ms | 1.59× |
| Top-1 prediction | " Paris" ✓ | — | Correct factual answer |
| Logits corr vs CPU F32 | 0.993 | — | |

### Profiling Scope

Both AIR and IRON are profiled with host overhead included:

**IRON**: `transformer.forward` (2.437s total / 16 layers) is timed by `sys.setprofile()`. Each operator's `forward()` includes torch→numpy conversion, `write_buffer()` (memory-mapped BO, zero-copy), `run_runlist()` (BO sync + kernel launch + wait), `read_buffer()` (memory-mapped, zero-copy), and numpy→torch conversion.

**AIR**: "Total prefill" (3.88s) is timed by `profiler.start_layer()` / `end_layer()` around `run_transformer_block()`. Each kernel's `load_and_run()` includes `bo.write()` + `bo.sync(TO_DEVICE)`, kernel launch + wait, `bo.sync(FROM_DEVICE)` + `bo.read()`, plus numpy reshaping and dtype conversion between kernel calls.

Both scopes include host overhead. IRON has lower host overhead due to memory-mapped BOs (zero-copy) vs AIR's explicit BO write/sync/read (3 copies per buffer).

**AIR also reports "kernel time" (3.11s)** which only covers the `load_and_run()` calls, excluding inter-kernel data prep. This has no IRON equivalent and understates the gap.

### Scope Outside 16 Transformer Layers

| Component | AIR | IRON | Notes |
|-----------|-----|------|-------|
| Token embedding | CPU lookup | CPU lookup | Both negligible |
| Final RMSNorm | NPU (in wall time) | NPU (in `model.forward`) | |
| **LM Head** | **CPU** (`np @ lm_head.T`) | **NPU** (`AIEGEMM 2048×2048×128256`) | AIR has no NPU LM Head yet |
| Weight loading | ~1.5s (in wall time) | Separate (before `model.forward`) | |

IRON `model.forward` = embedding + 16 layers + final norm + **NPU LM Head** = 2.744s.
AIR wall time 5.39s = weight loading (~1.5s) + embedding + 16 layers (3.88s) + final norm + CPU LM Head.

### Per-Block Breakdown (per layer)

| Block | AIR | IRON | Gap | Notes |
|-------|-----|------|-----|-------|
| Attention (QKV GEMMs + RoPE + Attn + O GEMM) | **65ms** | **76.9ms** | **0.85× (AIR faster)** | AIR FlashAttn 22ms vs IRON MHA 37ms |
| FFN (Gate + Up + SiLU×mul + Down) | **109ms** | **57.4ms** | **1.9× (IRON faster)** | IRON fuses all 4 ops into one kernel |
| RMSNorm ×2 | 16ms | 8.6ms | 1.9× | |
| Eltwise Add ×2 | 10ms | 9.0ms | 1.1× | |
| Host overhead (data prep between kernels) | ~43ms | ~0ms | — | AIR: numpy reshape/cast between each kernel |
| **Total per layer** | **243ms** | **152ms** | **1.59×** | |

Note: AIR per-block numbers come from "kernel time" profiling (inside `load_and_run`) and sum to ~200ms. The remaining ~43ms/layer is host-side data prep (numpy reshaping, dtype conversion, weight slicing) between kernel calls that IRON avoids via fused operators and PyTorch tensor views.

**Largest remaining gaps**:
1. **FFN** (109ms vs 57.4ms): IRON fuses Gate+Up+SiLU×mul+Down into one kernel, eliminating 3 DDR round-trips. AIR runs 4 separate kernels.
2. **Host overhead** (~43ms/layer): AIR does explicit BO write/sync/read + numpy reshaping between every kernel. IRON uses memory-mapped BOs and PyTorch tensor views.
3. **RMSNorm** (8ms vs 4.3ms): IRON is 1.9× faster.

**FlashAttention integrated** (2026-03-26): CPU attention fallback eliminated. Wall time dropped from ~44s to **5.39s** (8.2× improvement). NPU kernel total 3.11s (flash_attn 22ms avg/layer, 0.35s total for 16 layers).

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
| 7 | **Attention** | 32 heads, 2048 seq, causal | **15 (NPU)** | **37.0 (NPU)** | **2.5× faster** |
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
| **FlashAttention (causal)** | **15,022** | **30,989** | **2.06× faster** |
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
| **1** | **Integrate NPU FlashAttention** | **~38s wall** (replace CPU fallback) | **Ready (fixed 2026-03-26)** |
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

### AIR DMA Buffer Descriptor (BD) Exhaustion (upstream issue)

The ShimDMA hardware has **16 BD slots per channel**. Each BD can repeat up to 32 times, so one channel handles at most `16 BDs × 32 repeats × tile_n` elements. When the per-column data exceeds this, compilation fails.

| Buffer size | Per-column data | 16 BDs × 32 × tile_n=2048 | Fits? |
|------------|----------------|--------------------------|-------|
| 4.2M / 8 cols | 524K | 1,048K | **YES** |
| 16.8M / 8 cols | 2,097K | 1,048K | **NO** |
| 16.8M / 8 cols (tile_n=4096) | 2,097K | 2,097K | **YES** |

**Workaround**: Increase tile_n so that `16 × 32 × tile_n ≥ n / num_columns`.

**IRON avoids this** via ObjectFIFO, which uses repeating BD patterns with address auto-increment (2 BDs per buffer regardless of data size).

**Upstream fix needed**: AIR's `dma_memcpy_nd` lowering should generate repeating BD patterns instead of per-iteration BD chains. See `kernels/swiglu.md` for detailed analysis.

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
