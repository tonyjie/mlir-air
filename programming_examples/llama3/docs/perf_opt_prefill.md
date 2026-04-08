# LLAMA-3.2-1B Prefill — Performance Optimization

## Overview

Performance optimization of the LLAMA-3.2-1B BF16 prefill pipeline (seq_len=2048, 16 layers) on NPU2 (AIE2P).

Reference: IRON (`/home/jiajli/apps/IRON`). IRON profiling data: `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`.

---

## End-to-End Prefill Comparison

| Metric | AIR (current) | AIR (prev) | IRON | Notes |
|--------|--------------|------------|------|-------|
| **16 transformer layers** | **1.58s** | 1.71s | **2.44s** | **35% faster than IRON** |
| **LM Head** | **171ms** | 173ms | **217ms** | AIR 21% faster (8-launch ELF, static weight BOs) |
| **Total prefill** | **1.77s** | 2.05s | **2.744s** | **35% faster** (same scope: layers + norm + LM Head) |
| **Wall time** | **2.35s** | 2.51s | **2.75s** | AIR faster (includes weight loading) |
| Per-layer avg | **~100ms** | 107ms | 152ms | **0.66× (AIR faster)** |
| XRT invocations/layer | **5** | 5 | ~12 | All merges integrated |
| Top-1 prediction | " Paris" ✓ | — | — | Correct factual answer |
| Logits corr vs CPU F32 | 0.993 | 0.989 | — | |

### Profiling Scope

Both AIR and IRON are profiled with host overhead included:

**IRON**: `transformer.forward` (2.437s total / 16 layers) is timed by `sys.setprofile()`. Each operator's `forward()` includes torch→numpy conversion, `write_buffer()` (memory-mapped BO, zero-copy), `run_runlist()` (BO sync + kernel launch + wait), `read_buffer()` (memory-mapped, zero-copy), and numpy→torch conversion.

**AIR**: "Total prefill" (2.45s) is timed by `profiler.start_layer()` / `end_layer()` around `run_transformer_block()`. Each kernel's `load_and_run()` includes `bo.write()` + `bo.sync(TO_DEVICE)`, kernel launch + wait, `bo.sync(FROM_DEVICE)` + `bo.read()` (output only), plus numpy reshaping and dtype conversion between kernel calls. Both FFN and attention GEMMs use multi-launch (multiple launches in 1 ELF) with read-only-output optimization.

Both scopes include host overhead. See `host_optimization.md` for detailed BO write/read analysis.

**AIR also reports "kernel time" (1.93s)** which only covers the `load_and_run()` calls, excluding inter-kernel data prep.

### Scope Outside 16 Transformer Layers

| Component | AIR | IRON | Notes |
|-----------|-----|------|-------|
| Token embedding | CPU lookup | CPU lookup | Both negligible |
| Final RMSNorm | NPU (in wall time) | NPU (in `model.forward`) | |
| **LM Head** | **CPU** (`np @ lm_head.T`) | **NPU** (`AIEGEMM 2048×2048×128256`) | AIR has no NPU LM Head yet |
| Weight loading | ~1.5s (in wall time) | Separate (before `model.forward`) | |

IRON `model.forward` = embedding + 16 layers + final norm + NPU LM Head = 2.744s.
AIR total prefill = embedding + 16 layers + final norm + NPU LM Head = **1.77s**.

### Per-Block Breakdown (per layer)

| Block | AIR (current) | IRON | Notes |
|-------|--------------|------|-------|
| RMSNorm + QKV GEMMs | **9ms** | ~15ms | 4-launch ELF, 8-tile RMSNorm |
| RoPE Q+K | **4ms** | ~11ms | 2-herd ELF, 8-tile RoPE |
| FlashAttention | **22ms** | ~31ms | Seq-first ELF, strided DMA |
| O GEMM + Residual Add | **6ms** | ~15ms | 2-launch ELF |
| FFN Full (RMS+Gate+Up+SiLU+Down+Add) | **52ms** | **66ms** | 6-launch ELF, 8-tile RMSNorm |
| **Total per layer** | **~100ms** | **152ms** | **0.66× (AIR faster)** |

| Non-layer component | AIR | IRON | Notes |
|---------------------|-----|------|-------|
| LM Head | **171ms** | 217ms | 8-launch ELF, static weight BOs, bo.map() zero-copy |
| Final RMSNorm | 3ms | ~4ms | 8-tile herd |
| Embedding + overhead | ~50ms | ~86ms | |

**Key optimizations applied:**
- **Multi-launch ELF**: 5 XRT invocations/layer (down from 10), each ELF contains 2-6 `air.launch` ops
- **8-tile RMSNorm**: Broadcast weight DMA to 8 tiles (was 1-tile, now 6.7x faster standalone)
- **NPU LM Head**: 8-partition multi-launch ELF (single XRT call), pre-loaded weight BOs
- **`bo.map()` zero-copy**: All kernels use `bo.map()` for reads (like IRON), eliminating output memcpy
- **Static weight BOs**: LM Head weights written once at init, only `bo.sync()` per inference

**Remaining optimization opportunities:**
- **Plan B (DMA transpose)**: Would merge 5 → 3 invocations/layer. Blocked by AIE DMA stride=1 for BF16.
- **RMSNorm multi-tile** (8ms → ~4ms) — blocked by aiecc weight broadcast bug

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
**FFN multi-launch** (2026-03-30): 4 FFN kernels fused into 1 ELF.
**Read-only-output** (2026-03-31): Skip reading back intermediates/weights. FFN 83ms → 52ms. Total kernel 2.71s → 2.10s. Wall time 4.88s → 4.51s.

---

## Per-Kernel Comparison (per-invocation, in LLAMA pipeline)

AIR numbers include BO data transfer (write + sync + kernel + sync + read).
IRON model numbers from `operator.forward()` (includes write_buffer + run_runlist + read_buffer).

| Step | Kernel | Shape | AIR (ms) | IRON model (ms) | Gap |
|------|--------|-------|---------|-----------------|-----|
| 1,10 | RMSNorm | (2048, 2048) | 6.3 (C++ profiled) | 4.3 | 1.5× |
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
| RMSNorm (4M BF16) | **6,287** | 843* | **Single tile; multi-tile blocked** |
| RoPE Q (65536×64) | — | 845 | — |
| RoPE K (16384×64) | — | 257 | — |

AIR GEMM kernels are **23-32% faster** than IRON standalone. Eltwise add is matched.

---

## Next Steps

### Future Prefill Improvements

| Priority | Action | Estimated savings | Status | Details |
|----------|--------|-------------------|--------|---------|
| 1 | Multi-tile RMSNorm | ~4ms/layer (~50ms total) | **Blocked** | aiecc weight broadcast DMA bug: `stride=0` rejected. See `issues/github_issue_weight_broadcast_dma.md` |
| 2 | NPU transpose launches (5→3 invocations) | ~3-5ms/layer | **Blocked** | AIE DMA stride=1 for BF16. Needs C++ transpose kernel. See `issues/dma_transpose.py` |
| 3 | True FFN kernel fusion (single launch) | ~5-10ms/layer | **Future** | Fuse Gate+Up+SiLU×mul+Down into 1 AIR launch; enables L2-level data reuse |
| 4 | LM Head partition reduction (8→4) | ~50ms | **Future** | Larger N_part reduces dispatch overhead. Needs padding-aware GEMM or faster compilation for large N |

### Compiler Bugs Blocking Further Optimization

| Bug | Impact | Reproducer | Status |
|-----|--------|-----------|--------|
| `identifyLaunchRegions` only handles `SegmentLoadOp` | Bare `air.herd` launches silently dropped in multi-launch ELF | `issues/repro_herd_load_bug.py` | **Workaround applied** (wrap in `air.segment`) |
| Broadcast DMA generates `stride=0` in `aie.dma_bd` | Multi-tile herd with shared data (e.g. weight broadcast) fails | `issues/github_issue_weight_broadcast_dma.md` | **Upstream fix needed** |
| BF16 DMA innermost stride must be 1 | Cannot do DMA-only transpose for BF16 data | `issues/dma_transpose.py` | **Hardware limitation** — needs C++ kernel |

---

## Optimization History

### Totals Progression

| Metric | Baseline | +BF16 Add | +XRT Reuse | +BO Reuse | +GEMM/SwiGLU | +FlashAttn | +FFN Multi | +ReadOpt | +AttnGEMMs | +AllMerges | +LMHead+bomap | IRON |
|--------|----------|-----------|------------|-----------|-------------|------------|------------|---------|-----------|------------|---------------|------|
| Per-layer | 1.17s | 0.84s | 0.55s | 0.41s | 0.22s | 0.243s | 0.190s | 0.160s | 0.140s | 0.113s | **0.107s** | 0.152s |
| 16 layers | 18.67s | 13.40s | 8.77s | 6.49s | 3.57s | 3.88s | 3.25s | 2.65s | 2.45s | 1.81s | **1.71s** | 2.44s |
| Total prefill | — | — | — | — | — | — | — | — | — | — | **2.05s** | 2.744s |
| Wall time | 25.9s | 51.4s | ~47s | ~47s | ~44s | 5.39s | 4.88s | 4.51s | 4.00s | 3.89s | **2.51s** | 2.75s |
| XRT inv/layer | 15 | 15 | 15 | 15 | 15 | 15 | 12 | 12 | 10 | 5 | **5** | ~12 |
| vs IRON | 7.8× | 5.6× | 3.7× | 2.7× | 1.5× | 1.59× | 1.25× | 1.05× | 0.92× | 0.74× | **0.70×** | 1.0× |

+LMHead+bomap (2026-03-31): NPU LM Head (8-launch ELF, 173ms vs IRON 217ms). `bo.map()` zero-copy for all kernels. Static weight BO pre-loading. AIR 25% faster than IRON overall (2.05s vs 2.744s).

+8tileRMS (2026-04-02): 8-tile RMSNorm with broadcast weight DMA (bug fixed upstream). rms_attn_gemms 14→9ms, ffn_full 57→52ms, standalone rmsnorm 6→3ms. AIR 35% faster than IRON overall (1.92s vs 2.744s).

+8tileRoPE (2026-04-07): 8-tile RoPE (row-parallel, herd_x=8). rope_qk 11→4ms per layer. AIR 33% faster (1.84s vs 2.744s).

+seqfirst (2026-04-08): Seq-first RoPE + FlashAttention. Eliminated all 6 host-side transposes. RoPE uses repeat-ordered LUT; FlashAttention uses strided shim DMA for per-head extraction. **AIR 35% faster than IRON** overall (1.77s vs 2.744s).

### Per-Kernel Breakdown (current, per-invocation avg, ms)

| Kernel | Invocations/layer | Avg (ms) | Notes |
|--------|-------------------|----------|-------|
| rms_attn_gemms | ×1 | 9 | ELF, 4 launches (RMS[8-tile]+Q+K+V) |
| rope_qk | ×1 | 4 | ELF, 2 herds [8,1] (8-tile RoPE) |
| flash_attn | ×1 | 22 | Seq-first ELF, strided per-head DMA |
| o_proj_add | ×1 | 6 | ELF, 2 launches (Merge B) |
| ffn_full | ×1 | 52 | ELF, 6 launches (RMS[8-tile]+FFN+Add) |
| **Per-layer total** | **5** | **~100** | |
| lm_head | ×1 (total) | 171 | ELF, 8 launches, static weight BOs |

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

Fused Gate GEMM + Up GEMM + SiLU×mul + Down GEMM into 4 `air.launch` ops in a single ELF. One `xrt.run()` executes all 4 launches — intermediates flow through shared DDR buffers without host memcpy. FFN: 109ms → 83ms (24% faster). See `kernels/ffn_swiglu.md`.

### 8. Read-Only-Output (2026-03-31)

Only sync + read the last buffer (output) after kernel execution. Previously all buffers (including inputs, weights, intermediates) were read back — 218MB of unnecessary memcpy for the FFN block. FFN: 83ms → 52ms (37% faster). See `host_optimization.md`.

### 9. Attention GEMMs Multi-Launch (2026-03-31)

Fused Q+K+V GEMM projections into 3 `air.launch` ops in a single ELF. One `xrt.run()` for all 3 projections. QKV: 22ms → 8ms (2.8× faster). Per-layer: 160ms → 140ms. **AIR now faster than IRON per-layer (0.92×)**.

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
- `kernels/ffn_swiglu.md` — Multi-launch FFN block optimization, MLIR stitching
- `kernels/rmsnorm.md` — Multi-tile investigation, aiecc weight broadcast bug
