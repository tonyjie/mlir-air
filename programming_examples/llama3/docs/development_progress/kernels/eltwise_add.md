# Eltwise Add Kernel — Performance Analysis

> **Development history document.** This records kernel optimization analysis
> done during development. For the current pipeline, see `docs/profile.md`
> and `docs/explain.md`. Kernel names and invocation counts may have changed
> due to multi-launch merging.


## Role in LLAMA Pipeline

Residual additions in steps 9 and 15 of each transformer block:
- Step 9: `res1 = x + proj` (post-attention residual)
- Step 15: `output = res1 + down` (post-FFN residual)

**Invocations per forward pass**: 32 (2 per layer × 16 layers)

**Problem size**: n = seq_len × emb_dim = 2048 × 2048 = 4,194,304 elements

---

## Implementation History

### v1: F32 Scalar (baseline)

| Parameter | Value |
|-----------|-------|
| Data type | F32 (4 bytes/element) |
| Herd size | [1, 2] (2 cores) |
| Tile size | 1024 elements |
| Memory path | DDR → L1 direct (no L2) |
| Compute | Scalar `load → load → add → store` per element |
| Vectorization | None |

### v2: BF16 Vectorized (current)

| Parameter | Value |
|-----------|-------|
| Data type | BF16 (2 bytes/element) |
| Herd size | **[8, 1]** (8 cores across 8 AIE columns) |
| Tile size | 2048 elements |
| Memory path | DDR → L1 direct (no L2) |
| Compute | 16-wide `vector.transfer_read` → `arith.addf` → `vector.transfer_write` |
| Vectorization | 16 lanes (BF16 on AIE2P) |

---

## IRON Reference Implementation

**Source**: `/home/jiajli/apps/IRON/iron/operators/elementwise_add/`

| Parameter | Value |
|-----------|-------|
| Data type | BF16 (2 bytes/element) |
| Columns | 8 (one worker per AIE column) |
| Tile size | 2048 elements |
| Memory path | DDR → L1 direct (via ObjectFIFO) |
| Compute | 16-wide vectorized `aie::add` with pipelined II=1 |
| C++ kernel | `aie_kernels/generic/add.cc` — `eltwise_vadd<T_in, T_out>()` |

### IRON LLAMA config (prefill, 2048 tokens)

```python
# From iron/operators/elementwise_add/test.py
(4194304, 8, 2, 2048)  # size, num_aie_columns, num_channels, tile_size
```

### IRON correctness settings

| Setting | Value |
|---------|-------|
| Input range | `[0, 4)` uniform random, seed=42 |
| Input dtype | BF16 |
| Reference | PyTorch BF16 add (`output = input_a + input_b`) |
| Comparison space | BF16 → F32 per-element |
| Relative tolerance | **4%** (`rel_tol=0.04`) |
| Absolute tolerance | 1e-6 |
| Tolerance formula | `diff < max(abs_tol, rel_tol * (|a| + |b|))` |

---

## Correctness Verification (2026-03-16)

### Our BF16 vectorized kernel vs IRON reference settings

| Check | Result |
|-------|--------|
| **Python `compile-and-run` (N=4M, rtol=1%)** | **PASS** |
| **Python `compile-and-run` (N=64K, rtol=1%)** | **PASS** |
| **C++ harness (N=4M, herd [8,1])** | **PASS** — all 4,194,304 elements correct |
| **IRON tolerance (4% relative)** | **PASS** — 0 failures out of 4,194,304 elements |
| **Input range** | `[0, 4)` uniform — matches IRON |
| **Output range** | `[0.002, 8.0]` — expected for `a+b` where `a,b ∈ [0,4)` |

### Precision analysis

- BF16 ULP at value 4.0: 0.03125
- BF16 ULP at value 8.0: 0.0625
- Numpy BF16 add and F32-add-then-truncate produce **identical** results (0 mismatches out of 4M elements)
- Our NPU kernel passes at **1% relative tolerance**, which is 4× tighter than IRON's 4% threshold
- The small diffs (~0.015-0.03) observed in initial tests were from Python scalar BF16 rounding in the reference computation, not from the NPU kernel

### Commands

```bash
cd programming_examples/eltwise_add

# Correctness (compile-and-run with Python verification, NPU2 defaults)
make run

# C++ profiling with correctness check (NPU2 defaults: BF16 vec16 [8,1], N=4M)
make profile

# NPU1 config (F32 scalar [1,2])
make run AIE_TARGET=aie2

# Override any parameter
make profile N=65536                    # smaller problem
make profile HERD_X=4 HERD_Y=1         # different herd
make run DTYPE=f32 VECTOR_SIZE=0        # F32 scalar mode

# LIT tests (from build/ directory)
lit -v programming_examples/eltwise_add/run_makefile_peano_npu2.lit   # NPU2: run + profile
lit -v programming_examples/eltwise_add/run_makefile_peano.lit        # NPU1: F32 scalar
```

---

## Profiling Results (2026-03-16)

### C++ Profiling Harness

**Harness**: `programming_examples/eltwise_add/test.cpp` (10 warmup + 20 measured iterations, XRT context reused)

### v1: F32 Scalar (baseline)

| Problem Size | Avg Latency | Bandwidth | Notes |
|-------------|-------------|-----------|-------|
| 4,194,304 (LLAMA) | **214,619 µs** | 0.23 GB/s | F32, [1,2] herd, scalar |
| 65,536 (default) | 3,415 µs | 0.23 GB/s | Same bandwidth — scalar-bound |

### v2: BF16 Vectorized — Best Config [8,1]

| Problem Size | Avg Latency | Bandwidth | Notes |
|-------------|-------------|-----------|-------|
| 4,194,304 (LLAMA) | **415 µs** | 60.6 GB/s | BF16, [8,1] herd, vec16 |
| 4,194,304 (min) | 392 µs | 64.2 GB/s | Peak performance |

### IRON Reference

| Problem Size | Avg Latency | Bandwidth | Notes |
|-------------|-------------|-----------|-------|
| 4,194,304 (LLAMA 2048tok) | **432 µs** | 57.6 GB/s | BF16, 8 columns, vec16 |
| 26,624 (LLAMA 13tok) | 53 µs | 3.7 GB/s | |

---

## Herd Configuration Sweep (n=4,194,304, BF16, tile_n=2048, vec16)

NPU2 has **8 columns × 4 rows** of compute tiles. We swept all herd `[hx, hy]` combinations with hx ∈ {1,2,4,8} and hy ∈ {1,2,4} (NPU2 has max 4 rows per column).

| Herd | Cores | Status | Avg Latency (µs) | BW (GB/s) | Min Latency (µs) | Peak BW (GB/s) | vs IRON | Error |
|------|-------|--------|-------------------|-----------|-------------------|----------------|---------|-------|
| [1,1] | 1 | PASS | 2,401 | 10.5 | 2,385 | 10.6 | 5.6× | — |
| [1,2] | 2 | PASS | 2,400 | 10.5 | 2,380 | 10.6 | 5.6× | — |
| [1,4] | 4 | PASS | 644 | 39.1 | 630 | 39.9 | 1.5× | — |
| [2,1] | 2 | PASS | 654 | 38.5 | 639 | 39.4 | 1.5× | — |
| [2,2] | 4 | PASS | 648 | 38.8 | 641 | 39.3 | 1.5× | — |
| [2,4] | 8 | **FAIL** | — | — | — | — | — | ShimDMA channel exhaustion |
| [4,1] | 4 | PASS | 641 | 39.3 | 630 | 39.9 | 1.5× | — |
| [4,2] | 8 | **FAIL** | — | — | — | — | — | ShimDMA channel exhaustion |
| [4,4] | 16 | **FAIL** | — | — | — | — | — | ShimDMA channel exhaustion |
| **[8,1]** | **8** | **PASS** | **415** | **60.6** | **392** | **64.2** | **0.96×** | — |
| [8,2] | 16 | **FAIL** | — | — | — | — | — | ShimDMA channel exhaustion |
| [8,4] | 32 | **FAIL** | — | — | — | — | — | ShimDMA channel exhaustion |

**Failure details**:
- **ShimDMA channel exhaustion**: `'air.channel.put' op failed to map to shim dma channels: out of channels`. Occurs when total tiles ≥ 8 with herd_y > 1. The compiler can't allocate enough DMA channels for 3 operands × many tiles across multiple columns. IRON avoids this by using ObjectFIFO which has more efficient DMA channel scheduling.

**Note on MLIR dumps**: All 12 configs have MLIR saved to `build_peano/mlir_dumps/herd_{hx}x{hy}.mlir`. The `--print-module-only` flag prints the AIR MLIR generated by Python *before* compilation. Compilation failures happen downstream in `aircc` (AIR → AIE → NPU instruction lowering), so the IR generation always succeeds.

### Key observations

1. **`[8,1]` is the only 8-core config that compiles** — 415 µs / 60.6 GB/s, **4% faster than IRON** (432 µs). Uses 8 columns × 1 row, each column with its own ShimDMA channel for maximum memory bandwidth.

2. **`[1,2]` shows no speedup over `[1,1]`**: Both at ~2,400 µs. When tiles are in the same column (hy > 1), they share a single ShimDMA channel, so the second tile adds no memory bandwidth. The kernel is memory-bound, not compute-bound.

3. **4-core configs plateau at ~640 µs** regardless of herd shape: [1,4], [2,1], [2,2], [4,1] all converge. This suggests 4 effective ShimDMA channels at these herd shapes.

4. **All configs with herd_y > 1 and total ≥ 8 tiles fail**: [2,4], [4,2], [4,4], [8,2], [8,4]. The multi-row multi-column pattern exhausts ShimDMA channels because each tile needs separate DMA descriptors for 3 operands.

5. **Only single-row herds scale to 8 columns**: `[8,1]` is the sweet spot — one tile per column maximizes bandwidth without exhausting DMA resources.

### Scaling analysis

| Effective columns | Configs | Latency range | BW range |
|-------------------|---------|---------------|----------|
| 1 | [1,1], [1,2] | 2,380-2,401 µs | 10.5 GB/s |
| ~4 | [1,4], [2,1], [2,2], [4,1] | 630-654 µs | 38.5-39.9 GB/s |
| 8 | [8,1] | 392-415 µs | 60.6-64.2 GB/s |

Bandwidth scales almost linearly with effective column count:
- 1 col → 10.5 GB/s
- 4 cols → 39.3 GB/s (3.7× vs 4× ideal)
- 8 cols → 60.6 GB/s (5.8× vs 8× ideal, 77% efficiency)

---

## Generated MLIR — Best Config `[8,1]`

Saved to `build_peano/mlir_dumps/herd_8x1.mlir`:

```mlir
#map = affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2) * 2048)>
module {
  func.func @eltwise_add(%arg0: memref<4194304xbf16>,
                         %arg1: memref<4194304xbf16>,
                         %arg2: memref<4194304xbf16>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c8, %sy=%c1)
        args(%a=%arg0, %b=%arg1, %c=%arg2)
        : memref<4194304xbf16>, memref<4194304xbf16>, memref<4194304xbf16> {

      // L1 allocation: 3 buffers × 2048 BF16 = 12 KB per core (of 64 KB)
      %l1_a   = memref.alloc() : memref<2048xbf16, 2 : i32>
      %l1_b   = memref.alloc() : memref<2048xbf16, 2 : i32>
      %l1_out = memref.alloc() : memref<2048xbf16, 2 : i32>

      // Outer loop: 256 iterations (4M / (2048 × 8 tiles))
      scf.for %iv = 0 to 4194304 step 16384 {
        %offset = affine.apply #map()[%iv, %tx, %ty]  // iv + tx * 2048

        // DMA: DDR → L1 (2048 BF16 elements = 4 KB per transfer)
        air.dma_memcpy_nd (%l1_a, %a[%offset] [2048] [1])
        air.dma_memcpy_nd (%l1_b, %b[%offset] [2048] [1])

        // Vectorized compute: 128 iterations (2048 / 16)
        scf.for %j = 0 to 2048 step 16 {
          %sub_a = memref.subview %l1_a[%j] [16] [1]
          %sub_b = memref.subview %l1_b[%j] [16] [1]
          %sub_c = memref.subview %l1_out[%j] [16] [1]

          %v_a = vector.transfer_read %sub_a[0]  : vector<16xbf16>
          %v_b = vector.transfer_read %sub_b[0]  : vector<16xbf16>
          %v_c = arith.addf %v_a, %v_b           : vector<16xbf16>
          vector.transfer_write %v_c, %sub_c[0]  : vector<16xbf16>
        }

        // DMA: L1 → DDR
        air.dma_memcpy_nd (%c[%offset] [2048] [1], %l1_out)
      }
    }
    return
  }
}
```

All herd config MLIRs are saved in `build_peano/mlir_dumps/herd_{hx}x{hy}.mlir`.

---

## Performance Progression (n=4,194,304)

| Version | Config | Latency | Bandwidth | Speedup vs v1 | Gap vs IRON |
|---------|--------|---------|-----------|---------------|-------------|
| **v1** | F32 scalar, [1,2] | 214,619 µs | 0.23 GB/s | 1× (baseline) | 497× |
| **v2** | BF16 vec16, [1,2] | 1,323 µs | 19.0 GB/s | 162× | 3.1× |
| **v2** | BF16 vec16, [1,4] | 644 µs | 39.1 GB/s | 333× | 1.5× |
| **v2** | BF16 vec16, [8,1] | **415 µs** | **60.6 GB/s** | **517×** | **0.96×** |
| IRON | BF16 vec16, 8 cols | 432 µs | 57.6 GB/s | 497× | 1.0× |

**v2 with [8,1] matches IRON** — actually 4% faster on average.

---

## Impact on LLAMA Prefill

### Standalone profiling (C++ harness, XRT context reused)

| Metric | v1 (F32 scalar) | v2 [8,1] (BF16 vec16) | Improvement |
|--------|-----------------|----------------------|-------------|
| Per-invocation | 214.6 ms | 0.42 ms | **517×** |
| Total (32 invocations) | 8.19s | 0.013s | **630×** |

### Integrated in LLAMA pipeline (Python, per-invocation XRT load/unload)

| Metric | v1 (F32 scalar) | v2 [8,1] (BF16 vec16) | Improvement |
|--------|-----------------|----------------------|-------------|
| Per-invocation (avg) | 0.256s | 0.042s | **6.1×** |
| Total (32 invocations) | 8.19s | 1.34s | **6.1×** |
| % of NPU kernel time | 44% | 10% | No longer bottleneck |
| NPU kernel total | 18.67s | 13.40s | **28% reduction** |

Note: The 6.1× improvement in-pipeline is less than the 517× standalone improvement because each invocation in the pipeline pays ~40ms of XRT load/unload overhead (dominating the ~0.4ms actual kernel time). The standalone C++ harness amortizes this overhead across 30 iterations.

### Correctness after integration

| Check | Result |
|-------|--------|
| 1-layer `--verify` | PASS — all 15 steps `[OK]`, top-1 matches CPU |
| 16-layer `--verify` | PASS — all 240 steps `[OK]`, top-1 = " Paris" (correct) |
| Logits correlation | 0.969 (vs 0.972 with F32 add — negligible degradation) |

### Current LLAMA profiling breakdown (16 layers, CPU attention + BF16 add)

| Kernel | Avg | Count | Total | % of NPU |
|--------|-----|-------|-------|----------|
| GEMM Gate/Up | 0.117s | x32 | 3.74s | **28%** |
| GEMM Q/O | 0.054s | x32 | 1.73s | 13% |
| GEMM Down | 0.102s | x16 | 1.63s | 12% |
| SwiGLU | 0.089s | x16 | 1.42s | 11% |
| Eltwise Add | 0.042s | x32 | 1.34s | 10% |
| GEMM K/V | 0.040s | x32 | 1.28s | 10% |
| RMSNorm | 0.030s | x33 | 0.99s | 7% |
| RoPE Q | 0.056s | x16 | 0.90s | 7% |
| RoPE K | 0.021s | x16 | 0.34s | 3% |
| **NPU Total** | | **225** | **13.40s** | |

---

## Next Steps

1. **Address XRT load/unload overhead**: Each invocation pays ~40ms overhead — amortizing this (context reuse) would reduce the add total from 1.34s to ~0.013s
2. **Move to next bottleneck kernel**: GEMM Gate/Up (28%) is now the largest contributor — profile standalone and compare with IRON
3. **Fix NPU flash attention**: CPU attention dominates wall time (74%); fixing the upstream kernel bug would dramatically improve total latency
