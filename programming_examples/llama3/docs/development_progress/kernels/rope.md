# RoPE (Rotary Position Embeddings) Kernel — Analysis

> **Development history document.** This records kernel optimization analysis
> done during development. For the current pipeline, see `docs/profile.md`
> and `docs/explain.md`. Kernel names and invocation counts may have changed
> due to multi-launch merging.


## Formula

RoPE encodes position by rotating pairs of elements in each head:

```
For input x of shape (num_rows, head_dim) and precomputed LUT:

  y[r, 2i]   = x[r, 2i]   * cos(θ_r,i) - x[r, 2i+1] * sin(θ_r,i)
  y[r, 2i+1] = x[r, 2i]   * sin(θ_r,i) + x[r, 2i+1] * cos(θ_r,i)

where θ_r,i = r / (10000^(2i / head_dim))
```

The cos/sin values are precomputed on the host into a LUT with interleaved `[cos, sin, cos, sin, ...]` layout per row. The kernel does element-wise multiply-add — no trig at runtime.

Each row is one (head, position) pair. Rows are completely independent — no cross-row dependencies.

---

## Implementation: Two Layers

### Layer 1: C++ Kernel (`rope.cc`) — Processes One Row

**File**: `programming_examples/rope_lut/rope.cc` (also at `mlir-aie/aie_kernels/aie2p/rope.cc`)

```cpp
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
    // dims = head_dim = 64
    for (int v = 0; v < dims; v += 16) {        // vectorized, 16 elements at a time
        x      = load_v<16>(input + v);          // [x0, x1, x2, x3, ...]
        cache  = load_v<16>(lut + v);            // [cos0, sin0, cos1, sin1, ...]

        x_even  = filter_even(x);                // [x0, x2, x4, ...]
        x_odd   = filter_odd(x);                 // [x1, x3, x5, ...]
        cos_val = filter_even(cache);            // [cos0, cos1, cos2, ...]
        sin_val = filter_odd(cache);             // [sin0, sin1, sin2, ...]

        out_even = x_even * cos - x_odd * sin;
        out_odd  = x_even * sin + x_odd * cos;

        y = interleave_zip(out_even, out_odd);   // [e0, o0, e1, o1, ...]
        store_v(output + v, y);
    }
}
```

This processes exactly **one row of `head_dim=64` elements** on a single AIE tile. It uses AIE vector intrinsics (`filter_even/odd`, `interleave_zip`) for efficient BF16 computation. The kernel has no concept of multiple rows or tiles.

**Compiled to**: `rope.o` (precompiled, copied to build directory at runtime)

### Layer 2: AIR Wrapper (`rope_lut.py`) — DMA Loop + Herd

**File**: `programming_examples/rope_lut/rope_lut.py`

```python
build_module(seq_len, embed_dim, np_dtype_in)
# seq_len = number of rows (NOT the transformer sequence length!)
# embed_dim = head_dim = 64
# total = seq_len * embed_dim
```

```python
@herd(name="herd_0", sizes=[1, 1], operands=[l3_in, l3_lut, l3_out])
def herd_body(_tx, _ty, ...):
    l1_in  = alloc(embed_dim)      # L1 buffer for one row
    l1_lut = alloc(embed_dim)      # L1 buffer for one LUT row
    l1_out = alloc(embed_dim)      # L1 buffer for output row

    for row_offset in range_(0, total, embed_dim):    # loops num_rows times
        DMA: L3[row_offset : row_offset+64] → L1 (input)
        DMA: L3[row_offset : row_offset+64] → L1 (lut)
        call rope(l1_in, l1_lut, l1_out, 64)          # C++ kernel
        DMA: L1 → L3[row_offset : row_offset+64]      # output
```

The AIR wrapper is the **loop driver**: it DMAs one row at a time between L3 and L1, calls the C++ kernel, and writes the result back. The single [1,1] herd processes all rows sequentially.

---

## Shapes in LLAMA-3.2-1B Pipeline

### How Q/K get reshaped before RoPE

The GEMM/GEMV outputs are in **(seq_len, emb_dim)** layout. RoPE needs **(num_rows, head_dim)** where each row is one (head, position) pair:

**Prefill reshape (before RoPE Q)**:
```
Q from GEMM:  (2048, 2048)                     ← seq-first, heads interleaved
  reshape  →  (2048, 32, 64)                    ← split into 32 heads
  transpose → (32, 2048, 64)                    ← head-first
  reshape  →  (65536, 64)                       ← flatten to 2D: 65536 rows
  flatten  →  (4194304,)                        ← 1D for kernel memref
```

**Decode reshape (before RoPE Q)**:
```
Q from GEMV:  (2048,)                           ← single token
  reshape  →  (32, 64)                          ← 32 heads of 64 elements
  flatten  →  (2048,)                           ← 1D for kernel memref
```

### Shapes passed to `build_module()`

| Context | `seq_len` param | Actual meaning | Rows | head_dim | Total elements | Loop iterations |
|---------|----------------|----------------|------|----------|---------------|-----------------|
| **Decode RoPE Q** | 32 | n_heads | 32 | 64 | 2,048 | 32 |
| **Decode RoPE K** | 8 | n_kv_heads | 8 | 64 | 512 | 8 |
| **Prefill RoPE Q** | 65,536 | n_heads × seq_len | 65,536 | 64 | 4,194,304 | 65,536 |
| **Prefill RoPE K** | 16,384 | n_kv_heads × seq_len | 16,384 | 64 | 1,048,576 | 16,384 |

Note: the `seq_len` parameter of `build_module()` is the number of **rows**, not the transformer sequence length. For prefill, it's `n_heads × transformer_seq_len` because each (head, position) pair is one row.

### How RoPE is called in the pipeline

**Decode** — two separate xclbin kernels:

```python
# llama3_decode.py, compile_decode_kernels()
cache.compile_and_cache("rope_q", build_rope(n_heads=32,     head_dim=64, bfloat16), ...)
cache.compile_and_cache("rope_k", build_rope(n_kv_heads=8,   head_dim=64, bfloat16), ...)

# run_decode_block() — 2 XRT invocations
_run("rope_q", ..., q_heads.flatten(), lut_q, q_roped_out)    # 32 rows
_run("rope_k", ..., k_heads.flatten(), lut_k, k_roped_out)    # 8 rows
```

**Prefill** — two herds stitched into one ELF via `rope_qk_multi.py`:

```python
# multi_launch_builder/rope_qk_multi.py, build_rope_qk_module()
q_ir = str(build_rope(N_Q=65536, head_dim=64, bfloat16))     # 65536 rows
k_ir = str(build_rope(N_K=16384, head_dim=64, bfloat16))     # 16384 rows
# → Text-stitch into one func @rope_qk with 6 args, 2 sequential herds

# run_transformer_block() — 1 XRT invocation
_run_cached(cache, "rope_qk", ..., q_in, lut_q_in, k_in, lut_k_in, q_out, k_out)
```

---

## LLAMA-3.2-1B Configurations

### Config summary

| Usage | Kernel | Shape (rows × dim) | herd | Standalone (C++) | In pipeline | Rationale |
|-------|--------|-------------------|------|-----------------|-------------|-----------|
| **Prefill Q** | `rope_qk` ELF | 65536 × 64 | **[8,1]** | **912 us** | ~2ms | 8 tiles saturate bandwidth at 27.6 GB/s |
| **Prefill K** | `rope_qk` ELF | 16384 × 64 | **[8,1]** | **268 us** | ~1ms | Same kernel, stitched via multi-launch |
| **Decode Q** | `rope_q` xclbin | 32 × 64 | **[1,1]** | **49 us** | ~0.3ms | Too few rows; [8,1] is 27% slower (62us) |
| **Decode K** | `rope_k` xclbin | 8 × 64 | **[1,1]** | **54 us** | ~0.2ms | Too few rows; [8,1] is 6% slower (57us) |

### Prefill: why [8,1]

Each of the 65,536 rows is independent — perfect row-parallel. With 8 tiles, each processes 8,192 rows.

```python
# multi_launch_builder/rope_qk_multi.py
q_ir = str(build_rope(N_Q, head_dim, bfloat16, herd_x=8))
k_ir = str(build_rope(N_K, head_dim, bfloat16, herd_x=8))
```

**Profiling (C++ harness, 10 warmup + 20 measured):**

| Shape | herd=[1,1] | herd=[8,1] | Speedup | Bandwidth |
|-------|-----------|-----------|---------|-----------|
| Q: 65536 × 64 | 6717 us | **912 us** | **7.4x** | 27.6 GB/s |
| K: 16384 × 64 | 1716 us | **268 us** | **6.4x** | 23.5 GB/s |
| Combined Q+K | 8433 us | **1181 us** | **7.1x** | — |

Correctness: correlation = 0.999992 at all shapes and herd sizes.

**In-pipeline impact** (from `llama3_prefill.py --profile`):
- rope_qk per layer: 11ms → **4ms** (2.75x faster)
- 16 layers: 176ms → **64ms** (112ms saved)
- Total prefill: 1.92s → **1.84s**

### Decode: why [1,1]

**Profiling (C++ harness, 10 warmup + 20 measured):**

| Shape | herd=[1,1] (min/avg) | herd=[8,1] (min/avg) | Overhead |
|-------|---------------------|---------------------|----------|
| Q: 32 × 64 | **49 / 54 us** | 62 / 62 us | **+27% slower** |
| K: 8 × 64 | **54 / 54 us** | 57 / 58 us | **+6% slower** |

With only 32/8 rows, the 8-tile herd setup overhead (configuring DMAs for 8 tiles, synchronization) exceeds any compute savings. The kernel dispatch time (~50us) dominates at this scale — the actual row processing is negligible.

```python
# llama3_decode.py, compile_decode_kernels()
cache.compile_and_cache("rope_q", build_rope(n_heads=32,   head_dim=64, bfloat16))  # herd_x=1 default
cache.compile_and_cache("rope_k", build_rope(n_kv_heads=8, head_dim=64, bfloat16))  # herd_x=1 default
```

### Compile-time parameters

| Parameter | Prefill Q | Prefill K | Decode Q | Decode K |
|-----------|----------|----------|---------|---------|
| `seq_len` (rows) | 65536 | 16384 | 32 | 8 |
| `embed_dim` | 64 | 64 | 64 | 64 |
| `herd_x` | 8 | 8 | 1 | 1 |
| `herd_y` | 1 | 1 | 1 | 1 |
| `rows_per_tile` | 8192 | 2048 | 32 | 8 |
| Output format | ELF (multi-launch) | ELF (multi-launch) | xclbin | xclbin |
| Backend flags | `runtime_loop_tiling_sizes=[4,4]` | same | same | same |

---

## Multi-Tile Architecture

Each row's RoPE is completely independent — row `r` only reads `x[r,:]` and `lut[r,:]`. No cross-row dependency, no shared data to broadcast.

The `rope.cc` C++ kernel is unchanged — it processes one row at a time. Only the AIR wrapper (`rope_lut.py`) distributes rows across tiles:

```python
@herd(name="herd_0", sizes=[herd_x, 1], operands=[...])
def herd_body(_tx, _ty, ...):
    rows_per_tile = seq_len // herd_x                      # 65536/8 = 8192
    for local_row in range_(rows_per_tile):                 # 8192 iterations per tile
        row_offset = affine_apply(                          # tile-dependent offset
            (local_row + _tx * rows_per_tile) * embed_dim)
        DMA input[row_offset] → L1
        DMA lut[row_offset] → L1
        call rope(l1_in, l1_lut, l1_out, embed_dim)
        DMA L1 → output[row_offset]
```

Unified code path for all herd sizes (herd_x=1 is just the general case with `rows_per_tile = seq_len`). No separate single-tile branch needed.

### Profiling infrastructure

```bash
cd programming_examples/rope_lut

# Correctness test
make run SEQ_LEN=65536 EMBED_DIM=64 HERD_X=8

# C++ profiling (10 warmup + 20 measured, microsecond precision)
make profile SEQ_LEN=65536 EMBED_DIM=64 HERD_X=8

# LLAMA shapes: correctness + profiling for both Q and K
make run_llama
```

### Integration into prefill pipeline

The prefill uses `rope_qk_multi.py` which stitches two `build_rope()` calls via text-based MLIR. To enable multi-tile:

```python
q_ir = str(build_rope(N_Q, head_dim, bfloat16, herd_x=8))    # was herd_x=1
k_ir = str(build_rope(N_K, head_dim, bfloat16, herd_x=8))
```

No changes needed to the multi-launch stitching logic.

---

## 2D Scaling Investigation (herd_x × herd_y)

We investigated using 2D herds (e.g. [8,4] = 32 tiles) to push beyond the 8-tile [8,1] configuration. NPU2 has 8 columns × 4 compute rows = 32 tiles total.

### Results summary

| Config | Path | Tiles | LLAMA Q (65536x64) | Small (512x64) |
|--------|------|-------|--------------------|----------------|
| [1,1] | Direct | 1 | 6716 us | — |
| **[8,1]** | **Direct** | **8** | **912 us (7.4x)** | **67 us** |
| [4,2] | Direct | 8 | 912 us | 68 us |
| [8,2] | Direct | 16 | FAIL: out of shim DMA channels | — |
| [8,4] | Direct | 32 | FAIL: out of shim DMA channels | — |
| [8,2] | L2 tiled | 16 | FAIL: outer loop bug (only 1st batch) | 59 us (single batch) |
| [4,4] | L2 tiled | 16 | FAIL: outer loop bug | 55 us (single batch) |
| [8,4] | L2 tiled | 32 | FAIL: zeros even at small scale | FAIL |

### Blockers for >8 tiles

**1. Direct path (DDR→L1): shim DMA channel exhaustion**

Each tile needs 3 DMA channels (input, LUT, output). With >8 tiles, the shim DMA channels (limited per column) are exhausted. The `air-to-aie` pass reports: `'air.channel.put' op failed to map to shim dma channels: out of channels`.

**2. L2 path: segment-level outer loop only executes first iteration**

L2 staging (DDR→L2→L1) is needed to reduce shim channel pressure. But the tiled outer loop (`scf.for` at segment level wrapping a `herd`) only produces correct output for the first batch — subsequent batches output zeros. This appears to be a runtime/compiler issue with repeated herd invocations inside a segment-level loop.

Verified: single-batch L2 PASS, 2-batch L2 FAIL (same data, same herd config).

**3. [8,4] herd (32 tiles): zeros even without outer loop**

At small scale with a single batch, [8,4] still produces zeros. This affects any config where `herd_x = 8` and `herd_y >= 3` (24+ tiles). Configs up to 16 tiles ([8,2], [4,4]) work correctly. Root cause unclear — may be MemTile lock or BD exhaustion at 24+ concurrent tile connections.

### Why GEMM works with [8,4] but RoPE doesn't

GEMM's L2 buffers are structured so each buffer's DMA depends on only **one** tile coordinate:
- A: `[herd_m, 1, tile_m, tile_k_l2]` — L2→L1 uses `_tx` only (A is broadcast across `_ty`)
- B: `[1, herd_n, tile_k_l2, tile_n]` — L2→L1 uses `_ty` only (B is broadcast across `_tx`)
- Result: 8 + 4 = 12 BDs per buffer at MemTile level

RoPE has no natural broadcasting — every tile reads unique data (unique row). Each DMA needs both `_tx` AND `_ty` → 32 BDs per buffer → overflow.

### Conclusion

**[8,1] direct at 912us is the practical optimum** for this kernel. The kernel is bandwidth-limited at ~27.6 GB/s with 8 tiles. L2 staging with 16 tiles shows 18% improvement at small scale (55us vs 67us) but is blocked at LLAMA scale by the outer loop bug. Filing this as a future compiler issue.

Experimental L2 tiled implementation: `rope_lut_l2_tiled.py` (kept for reference).

---

## Alternative: `rope_sincos/` (On-Chip Sin/Cos)

There is another RoPE implementation at `programming_examples/rope_sincos/` that computes sin/cos **on-chip** via Chebyshev polynomial approximation instead of using a precomputed LUT.

| Aspect | `rope_lut/` (ours) | `rope_sincos/` |
|--------|-------------------|----------------|
| Sin/cos | Precomputed host LUT, DMA'd per row | Computed on-chip (4th-order Chebyshev) |
| Input format | Separate Q, K buffers | Packed QKV per head (`3 × head_size`) |
| Herd | [1,1] (sequential rows) | [1, herd_n] (parallel across heads) |
| head_dim | Any (64 for LLAMA) | Hardcoded 48 (freq table size) |
| Compiler | Peano (our toolchain) | Chess-only (`::shuffle` intrinsics) |

**Not usable for LLAMA** due to: head_dim=48 hardcoded (LLAMA needs 64), Chess-only intrinsics (we use Peano), and packed QKV layout mismatch. The on-chip sin/cos idea could save LUT DMA bandwidth, but the real bottleneck is 65K rows on 1 tile — multi-tiling `rope_lut` is the higher-impact fix.

---

## Files

| File | Purpose |
|------|---------|
| `programming_examples/rope_lut/rope_lut.py` | AIR kernel builder (direct DDR→L1, herd_x) |
| `programming_examples/rope_lut/rope_lut_l2_tiled.py` | Experimental L2-staging version (2D herd, blocked at scale) |
| `programming_examples/rope_lut/rope.cc` | C++ kernel (one row, vectorized BF16) |
| `programming_examples/rope_lut/test.cpp` | C++ profiling harness (XRT, microsecond) |
| `programming_examples/rope_lut/Makefile` | Build: `run`, `profile`, `run_llama` targets |
| `programming_examples/rope_lut/run_llama_shape_peano.lit` | LIT test at LLAMA Q+K shapes |
| `llama3/multi_launch_builder/rope_qk_multi.py` | Prefill multi-launch: Q+K in one ELF |
| `llama3/llama3_decode.py` | Decode: compiles `rope_q` and `rope_k` separately |
| `llama3/llama3_prefill.py` | Prefill: calls `rope_qk_multi.build_rope_qk_module()` |
