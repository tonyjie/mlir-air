# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P).

Reference: IRON (`/home/jiajli/apps/IRON`). IRON profiling data: `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`.

---

## End-to-End Prefill Comparison

| Metric | AIR (current) | IRON | Notes |
|--------|--------------|------|-------|
| **16 transformer layers** | **3.25s** | **2.44s** | **Gap: 1.33×** (same profiling scope — see below) |
| **Wall time** | **4.88s** | **2.75s** | AIR includes weight loading (~1.5s) + CPU LM Head |
| Per-layer avg | 190ms | 152ms | 1.25× |
| Top-1 prediction | " Paris" ✓ | — | Correct factual answer |
| Logits corr vs CPU F32 | 0.993 | — | |

### Profiling Scope

Both AIR and IRON are profiled with host overhead included:

**IRON**: `transformer.forward` (2.437s total / 16 layers) is timed by `sys.setprofile()`. Each operator's `forward()` includes torch→numpy conversion, `write_buffer()` (memory-mapped BO, zero-copy), `run_runlist()` (BO sync + kernel launch + wait), `read_buffer()` (memory-mapped, zero-copy), and numpy→torch conversion.

**AIR**: "Total prefill" (3.25s) is timed by `profiler.start_layer()` / `end_layer()` around `run_transformer_block()`. Each kernel's `load_and_run()` includes `bo.write()` + `bo.sync(TO_DEVICE)`, kernel launch + wait, `bo.sync(FROM_DEVICE)` + `bo.read()`, plus numpy reshaping and dtype conversion between kernel calls. The FFN block uses multi-launch (4 launches in 1 ELF, single XRT invocation), reducing host overhead.

Both scopes include host overhead. IRON has lower host overhead due to memory-mapped BOs (zero-copy) vs AIR's explicit BO write/sync/read (3 copies per buffer).

**AIR also reports "kernel time" (2.71s)** which only covers the `load_and_run()` calls, excluding inter-kernel data prep. This has no IRON equivalent and understates the gap.

### Scope Outside 16 Transformer Layers

| Component | AIR | IRON | Notes |
|-----------|-----|------|-------|
| Token embedding | CPU lookup | CPU lookup | Both negligible |
| Final RMSNorm | NPU (in wall time) | NPU (in `model.forward`) | |
| **LM Head** | **CPU** (`np @ lm_head.T`) | **NPU** (`AIEGEMM 2048×2048×128256`) | AIR has no NPU LM Head yet |
| Weight loading | ~1.5s (in wall time) | Separate (before `model.forward`) | |

IRON `model.forward` = embedding + 16 layers + final norm + **NPU LM Head** = 2.744s.
AIR wall time 4.88s = weight loading (~1.5s) + embedding + 16 layers (3.25s) + final norm + CPU LM Head.

### Per-Block Breakdown (per layer)

| Block | AIR | IRON | Gap | Notes |
|-------|-----|------|-----|-------|
| Attention (QKV GEMMs + RoPE + Attn + O GEMM) | **65ms** | **76.9ms** | **0.85× (AIR faster)** | AIR FlashAttn 22ms vs IRON MHA 37ms |
| FFN (Gate + Up + SiLU×mul + Down) | **83ms** (multi-launch) | **57.4ms** | **1.4× (IRON faster)** | AIR multi-launch (4 launches in 1 ELF); was 109ms with separate kernels |
| RMSNorm ×2 | 16ms | 8.6ms | 1.9× | |
| Eltwise Add ×2 | 10ms | 9.0ms | 1.1× | |
| Host overhead (data prep between kernels) | ~21ms | ~0ms | — | Reduced from ~43ms by FFN multi-launch |
| **Total per layer** | **190ms** | **152ms** | **1.25×** | |

Note: FFN multi-launch eliminated ~26ms/layer of host overhead by combining 4 separate kernel calls into 1. Remaining host overhead (~21ms) is from attention-path kernels (QKV GEMMs, RoPE, FlashAttn, O GEMM, residual adds).

**Largest remaining gaps:**
- **FFN block** (83ms vs 57.4ms) — 26ms gap (reduced from 52ms by multi-launch)
- **Host overhead** (~21ms/layer) — remaining inter-kernel data prep for attention path
- **RMSNorm** (9ms vs 4.3ms) — 5ms gap

#### Understanding the host overhead on shared-memory Ryzen AI

CPU and NPU share the same DDR — there is no discrete device memory. XRT buffer objects (BOs) are DDR allocations accessible by both. The `bo.sync()` calls are **CPU cache coherency operations** (flush/invalidate), not DMA transfers.

The actual overhead in AIR comes from **unnecessary DDR→DDR memcpy** between kernel calls:

```
Gate GEMM output: NPU writes to BO_gate_out (DDR address A)
                  bo.read() → memcpy DDR(A) → numpy array (DDR address B)    # ~1-2ms for 33MB
                  ... numpy reshaping ...
SiLU×mul input:   bo.write() → memcpy numpy (DDR address B) → BO_swiglu_in (DDR address C)  # ~1-2ms
                  bo.sync(TO_DEVICE) → flush CPU cache                       # ~0.1ms
```

The data never leaves DDR, but gets copied between different DDR regions (BO memory ↔ numpy array memory) via CPU memcpy.

**IRON avoids this** by passing the same BO to consecutive kernels in `run_runlist()`:
- Gate GEMM writes to `BO["left"]`, SiLU reads from `BO["left"]` — same DDR address, zero copies
- `write_buffer()` uses `bo.map()` + `np.copyto()` — one copy into mapped BO memory (vs AIR's `bo.write()` which copies into an internal buffer)
- `read_buffer()` uses `bo.map()` + `np.frombuffer()` — zero-copy view of BO memory (vs AIR's `bo.read()` which allocates + copies)

Per-layer host overhead breakdown (AIR, estimated):

| Source | Cost | Occurrences/layer | Total |
|--------|------|-------------------|-------|
| `bo.write()` memcpy (inputs) | ~1-2ms per 33MB buffer | ~15 buffers | ~15-20ms |
| `bo.read()` memcpy (outputs) | ~1-2ms per 33MB buffer | ~15 buffers | ~15-20ms |
| numpy reshape/cast between kernels | ~0.5ms each | ~15 | ~5-8ms |
| `bo.sync()` cache ops | ~0.1ms each | ~30 | ~3ms |
| **Total host overhead** | | | **~40ms/layer** |

**FlashAttention integrated** (2026-03-26): CPU attention fallback eliminated.
**FFN multi-launch integrated** (2026-03-30): 4 FFN kernels fused into 1 ELF. Wall time 5.39s → 4.88s, kernel time 3.11s → 2.71s.

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
| 11-14 | **FFN block (multi-launch)** | Gate+Up+SiLU×mul+Down | **83** | **57.4** (fused) | **1.4×** |

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

## Next Steps

| Priority | Action | Estimated savings | Notes |
|----------|--------|-------------------|-------|
| **1** | **Attention-path multi-launch** | **~15-20ms/layer** | Apply same multi-launch pattern to QKV GEMMs + RoPE + Attn + O GEMM |
| **2** | Vectorize RoPE / RMSNorm | ~0.10s total | RoPE: 10ms vs IRON 5.3ms; RMSNorm: 9ms vs IRON 4.3ms |
| **3** | Host-side `bo.map()` zero-copy reads | ~5-10ms/layer | Eliminate remaining DDR memcpy in attention path |
| **4** | True FFN kernel fusion (single launch) | ~0.3s total | Fuse Gate+Up+SiLU×mul+Down into 1 AIR launch; enables L2-level data reuse |

Priorities 1-2 completed: FFN multi-launch (done), FlashAttention integration (done).

---

## Optimization History

### Totals Progression

| Metric | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | +GEMM/SwiGLU Opt | +FlashAttn | +FFN Multi | IRON |
|--------|----------|-----------|------------|-----------|-----------------|------------|------------|------|
| Per-layer (with host) | 1.17s | 0.84s | 0.55s | 0.41s | 0.22s | 0.243s | **0.190s** | 0.152s |
| 16 layers (with host) | 18.67s | 13.40s | 8.77s | 6.49s | 3.57s | 3.88s | **3.25s** | 2.44s |
| Wall time | 25.9s | 51.4s | ~47s | ~47s | ~44s | 5.39s | **4.88s** | 2.75s |
| Gap to IRON (per-layer) | 7.8× | 5.6× | 3.7× | 2.7× | 1.5× | 1.59× | **1.25×** | 1.0× |

FFN multi-launch (2026-03-30): Fused Gate+Up+SiLU×mul+Down into 4 `air.launch` ops in 1 ELF. Per-layer 243ms → 190ms (22% improvement). Gap to IRON narrowed from 1.59× to **1.25×**.

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

[8,1] herd + tile_n=4096 + 16-wide vectors. 59ms → 37ms (1.6×). BD exhaustion workaround: larger tiles to reduce iteration count under BD limit. See `kernels/silu_and_mul.md`.

### 6. FlashAttention Integration (2026-03-26)

Replaced CPU attention fallback with NPU FlashAttention kernel. Kernel takes unscaled Q (scaling handled internally), 4 args (Q, K, V, Output), causal masking built-in. Wall time: ~44s → 5.39s. FlashAttention: 22ms avg/layer, corr=0.9976 standalone. See `kernels/flash_attention.md`.

### 7. FFN Multi-Launch (2026-03-30)

Fused Gate GEMM + Up GEMM + SiLU×mul + Down GEMM into 4 `air.launch` ops in a single ELF. One `xrt.run()` executes all 4 launches — intermediates flow through shared DDR buffers without host memcpy. FFN: 109ms → 83ms (24% faster). Per-layer: 243ms → 190ms (22% faster). Gap to IRON: 1.59× → 1.25×. See `kernels/ffn_swiglu.md`.

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

**Upstream fix needed**: AIR's `dma_memcpy_nd` lowering should generate repeating BD patterns instead of per-iteration BD chains. See `kernels/silu_and_mul.md` for detailed analysis.

---

## Profiling Commands

```bash
# AIR: Full 16-layer profiling (NPU attention, default)
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile

# AIR: With verification
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --profile

# AIR: With CPU attention fallback (for comparison)
python3 ../llama3_prefill.py --run-only --n-layers 16 --profile --cpu-attn

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
- `kernels/silu_and_mul.md` — [8,1] optimization, BD exhaustion analysis
- `kernels/flash_attention.md` — Correctness investigation, precision metrics, status
