# LLAMA GEMM Precision & Configuration Reference

Reference document for GEMM kernel precision analysis and configuration choices for LLAMA-3.2-1B on NPU2.

---

## 1. GEMM Configurations: Ours vs IRON

### IRON's GEMM Configuration for LLAMA-3.2-1B

Source: `/home/jiajli/apps/IRON/iron/applications/llama_3.2_1b/`

| Parameter | IRON Value | Our Value | Notes |
|-----------|-----------|-----------|-------|
| **tile_m** | 64 | 32 | IRON uses 2x larger M tile |
| **tile_k** | 64 | 64 (tile_k_l2) | Same |
| **tile_n** | 64 | 32 | IRON uses 2x larger N tile |
| **n_aie_cols** | 8 | 4 (herd_n) | IRON uses full NPU2 width (8 cols) |
| **n_aie_rows** | 4 | 4 (herd_m) | Same |
| **BFP16 emulation** | True (default) | True | Same (but IRON tests use False) |
| **prio_accuracy** | False (default) | N/A (we use F32 output) | IRON tests use True |
| **use_static_weight** | True | False | IRON pre-loads weights to avoid repeated transfers |
| **Output dtype** | bf16 (default) | f32 (our fix) | Both achieve F32 accumulation differently |

### IRON's Per-Projection GEMM Shapes

All use tile_m=64, tile_k=64, tile_n=64, n_aie_cols=8:

| Projection | M | K | N | # L2 tiles (K) | # Output tiles |
|-----------|------|------|------|-----------------|----------------|
| Q/O | 2048 | 2048 | 2048 | 32 | 32x32=1024 |
| K/V | 2048 | 2048 | 512 | 32 | 32x8=256 |
| Gate/Up | 2048 | 2048 | 8192 | 32 | 32x128=4096 |
| Down | 2048 | 8192 | 2048 | 128 | 32x32=1024 |

### Our Per-Projection GEMM Configuration

Using tile_m=32, tile_k_l2=64, tile_k_l1=32, tile_n=32, herd_m=4, herd_n=4:

| Projection | M | K | N | # L2 tiles (K) | Launch grid |
|-----------|------|------|------|-----------------|-------------|
| Q/O | 2048 | 2048 | 2048 | 32 | 16x16 |
| K/V | 2048 | 2048 | 512 | 32 | 16x4 |
| Gate/Up | 2048 | 2048 | 8192 | 32 | 16x64 |
| Down | 2048 | 8192 | 2048 | 128 | 16x16 |

### Key Differences

1. **Tile sizes**: IRON uses 64x64x64, we use 32x64x32. Larger tiles mean fewer DMA transfers and less overhead, but require more L1 memory per tile.

2. **AIE columns**: IRON uses all 8 columns of NPU2. We use 4 (our `herd_n=4`). Using 8 columns would double the parallelism for the N dimension.

3. **Static weights**: IRON pre-loads weight matrices so they don't need to be transferred for every invocation. We re-transfer weights every time.

4. **FFN**: IRON uses a fused SwiGLU operator (`use_aie_ffn_swiglu=true`) instead of separate Gate/Up/Down GEMMs. The FFN GEMMs are disabled in their default config.

---

## 2. Precision Analysis

### The Problem

The Down GEMM (K=8192) had per-step correlation of 0.976 with F32 reference. Over 16 layers, this compounded to produce incorrect model output.

### Root Cause: BF16 Output Truncation

The AIE2P mmul hardware accumulates in F32, but when the output buffer type is BF16, the F32 accumulator is truncated to BF16 at every write-back to the C buffer:

```
With BF16 C buffer:
  For each of ~256 inner iterations:
    C_bf16 = truncate(extend(C_bf16) + A_bf16 * B_bf16)
    └── precision lost at every iteration

With F32 C buffer:
  For each of ~256 inner iterations:
    C_f32 = C_f32 + A_bf16 * B_bf16
    └── full F32 precision preserved
  Final: truncate(C_f32) to BF16
    └── precision lost only ONCE
```

### Experimental Results

**Test A: NPU GEMM correlation vs K dimension** (random data, BF16 output)

| K | # Accumulations | NPU corr | Degradation from K=512 |
|---|-----------------|----------|----------------------|
| 512 | ~64 | 0.9998 | baseline |
| 1024 | ~128 | 0.9995 | -0.03% |
| 2048 | ~256 | 0.9984 | -0.14% |
| 4096 | ~512 | 0.9938 | -0.60% |
| 8192 | ~1024 | 0.9767 | -2.31% |

Correlation degrades monotonically with K because more accumulation steps = more BF16 truncation events.

**Test B: BF16 tile-boundary truncation simulation** (K=8192, CPU)

Simulated the tiling on CPU by truncating to BF16 at every `tile_k` boundary:

| tile_k | # Truncation points | Simulated corr |
|--------|--------------------|-----------------------|
| 32 | 256 | 0.9998 |
| 64 | 128 | 0.9999 |
| 256 | 32 | 0.99998 |
| 8192 | 1 | 0.999999 |

Even with 256 truncation points, the simulated corr is 0.9998 -- much better than the NPU's 0.976. This means the NPU has more frequent truncation than just at tile boundaries.

**Test C: NPU with different tile_k_l2** (K=8192)

| tile_k_l2 | NPU corr | max_err |
|-----------|----------|---------|
| 64 | 0.976396 | 7.3457 |
| 128 | 0.976396 | 7.3457 |
| 256 | 0.976396 | 7.3457 |

Identical results regardless of tile_k_l2. The truncation happens inside the L1 computation (at the vector.contract write-back level), not at L2 tile boundaries.

**Test D: F32 output** (K=8192, the fix)

| Output type | corr | max_err |
|-------------|------|---------|
| BF16 | 0.976 | 7.35 |
| **F32** | **0.9999** | **0.16** |

Changing C buffer to F32 eliminates all internal truncation. The 45x improvement in corr confirms the root cause.

**Single-layer verification with F32 output (all projections):**

| GEMM | K | BF16 out corr | F32 out corr | Improvement |
|------|---|--------------|-------------|-------------|
| Q/O proj | 2048 | 0.998 | 0.9999 | 10x |
| K/V proj | 2048 | 0.998 | 0.9999 | 10x |
| Gate/Up | 2048 | 0.998 | 0.9999 | 10x |
| **Down** | **8192** | **0.948** | **0.9998** | **52x** |

### Where Exactly Does Truncation Happen?

The vectorized GEMM kernel (after transform IR) has this structure inside each L1 tile:

```
// Herd body for compute (simplified):
c_subview = subview(l1_c, [tx, ty, ...])   // BF16 subview of C buffer

// Zero fill
fill(0.0, c_subview)                        // Write BF16 zeros

// K reduction loop (L2 level)
for k_l2 in range(0, K, tile_k_l2):
    // DMA A, B from L2 to L1

    // Inner K loop (L1 level)
    for k_l1 in range(0, tile_k_l2, tile_k_l1):
        // DMA A, B tiles within L1

        // Vectorized matmul (after transform):
        // 1. Read C from BF16 buffer -> extf to F32
        // 2. vector.contract (F32 accumulation over K=8 mmul)
        // 3. truncf F32 -> BF16
        // 4. Write C back to BF16 buffer
        //
        // The cast hoisting moves extf/truncf outside the
        // innermost loop, but they still execute at every
        // L1 iteration (every 32 K elements).
        block_matmul(a_l1, b_l1, c_subview)

// DMA C from L1 to L2
```

With BF16 C buffer: steps 1-4 truncate at every L1 iteration.
With F32 C buffer: steps 1-4 operate entirely in F32 (no truncation).

---

## 3. BFP16 (Block Floating Point 16)

### What Is It?

BFP16 groups values into blocks that share a single exponent. This allows the mmul unit to process 2x more data per cycle (8x8x8 instead of 4x8x8) but reduces precision.

| Mode | Flag | mmul dims | Throughput | Precision |
|------|------|-----------|------------|-----------|
| True BF16 | (no flag) | 4×8×8 = 256 MACs | 1x | Higher |
| BFP16 emulation | `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` | 8×8×8 = 512 MACs | **2x** | Lower |

### Precision Impact

With our F32 output fix, BFP16 causes only minor precision loss:
- F32 output + BFP16 on: corr = 0.9999
- Theoretical F32 output + true BF16: corr ≈ 0.99999

The BFP16 precision loss is negligible compared to the BF16 output truncation issue. It's a throughput vs precision tradeoff that's acceptable for LLM inference.

### IRON's Default

IRON defaults `emulate_bf16_mmul_with_bfp16=True` in the operator (throughput priority), but their tests use `False` (precision priority). For LLAMA inference, they keep the default (BFP16 on).

---

## 4. Performance Comparison

### Why Our GEMM is Slow

Our prefill takes ~8 minutes per layer (2+ hours for 16 layers). The bottleneck is **recompilation**, not execution:

| Phase | Time per kernel | Times per layer | Total |
|-------|----------------|-----------------|-------|
| `build_module()` | ~1s | 10 | 10s |
| Transform IR | ~0.5s | 6 (GEMMs only) | 3s |
| `aircc` compilation | ~20-30s | 10 | 200-300s |
| NPU execution | ~0.1-1s | 15 | 5-15s |

Each of our 10 unique kernel configs is recompiled from scratch for every layer. With 16 layers, that's 160 compilations. IRON compiles once and reuses.

### IRON's Performance Advantages

1. **Compile once**: Pre-compiled kernel archives (`.a` files) contain optimized machine code. No per-invocation compilation.

2. **8 AIE columns**: Uses full NPU2 width (8 cols vs our 4), doubling N-dimension parallelism.

3. **Larger tiles**: 64x64x64 vs our 32x64x32. Fewer DMA transfers, better compute/transfer ratio.

4. **Static weights**: Weights pre-loaded to device memory, not re-transferred every invocation.

5. **Fused FFN**: SwiGLU operator fuses Gate+Up+Activation+Down into one kernel, eliminating 3 host-NPU round trips.

6. **While-true loop**: The device runs a persistent loop processing all Q chunks internally, vs our compile-load-run-unload per invocation.

### Optimization Roadmap for Our GEMM

| Priority | Optimization | Expected Impact |
|----------|-------------|-----------------|
| 1 | **Cache compiled ELFs** -- avoid recompilation for same kernel config | ~10x faster (5 min -> 30s per layer) |
| 2 | **Use 8 AIE columns** (herd_n=8) | ~2x throughput |
| 3 | **Larger tiles** (64x64x64) | Better compute/transfer ratio |
| 4 | **Static weights** -- pre-load weight matrices | Reduce DMA overhead |
| 5 | **Fuse FFN** -- combine Gate+Up+SwiGLU+Down | Eliminate 3 round trips per layer |
| 6 | **Persistent device loop** -- `omit_while_true_loop=False` | Reduce host-device synchronization |

---

## 5. Recommended GEMM Configuration

Based on IRON's proven configuration and our precision investigation:

### For Accuracy (Current Priority)

```python
build_gemm(
    m, k, n,
    tile_m=32,         # Keep current (64 may exceed L1 with F32 output)
    tile_k_l2=64,      # Keep current
    tile_k_l1=32,      # Keep current
    tile_n=32,          # Keep current
    herd_m=4,
    herd_n=4,
    np_dtype_in=bfloat16,
    np_dtype_out=np.float32,  # F32 output for precision
    arch="aie2p",
    direct_codegen=True,
)
```

### For Performance (Future)

```python
# Match IRON's tile config:
tile_m=64, tile_k=64, tile_n=64
herd_m=4, herd_n=8  # Use full 8 AIE columns

# Or use IRON's kernel directly via external archive:
# lower_linalg_to_func="gemm_64x64x64_archive.a"
```

### L1 Memory Budget

| Config | A tile | B tile | C tile | Total | Fits 64KB? |
|--------|--------|--------|--------|-------|-----------|
| 32×64×32, BF16 out | 32×32×2=2KB | 32×32×2=2KB | 32×32×2=2KB | 6KB | Yes |
| 32×64×32, F32 out | 32×32×2=2KB | 32×32×2=2KB | 32×32×4=4KB | 8KB | Yes |
| 64×64×64, BF16 out | 64×64×2=8KB | 64×64×2=8KB | 64×64×2=8KB | 24KB | Yes |
| 64×64×64, F32 out | 64×64×2=8KB | 64×64×2=8KB | 64×64×4=16KB | 32KB | Yes |

All configurations fit within the 64KB L1 budget.

---

## Document References

- `LLAMA_PLAN.md` -- High-level plan
- `LLAMA_progress.md` -- Session log
- `LLAMA_verification.md` -- Test results, commands, bugs
- `LLAMA_explanation.md` -- Code walkthrough
- `LLAMA_gemm.md` -- This file (GEMM precision & config reference)
- IRON source: `/home/jiajli/apps/IRON/iron/operators/gemm/design.py`
- IRON LLAMA config: `/home/jiajli/apps/IRON/iron/applications/llama_3.2_1b/configs/llama32_1b.json`
