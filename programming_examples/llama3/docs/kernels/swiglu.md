# SwiGLU Activation Kernel — Performance Analysis

## Role in LLAMA Pipeline

Step 13 of each transformer block: `output = SiLU(gate) × up`

Where `SiLU(x) = x × sigmoid(x) = x × 0.5 × (tanh(x/2) + 1)`

| Parameter | Value |
|-----------|-------|
| Input size | seq_len × hidden_dim = 2048 × 8192 = 16,777,216 BF16 elements |
| Inputs | gate (from GEMM step 11), up (from GEMM step 12) |
| Output | SiLU(gate) × up |
| Invocations per layer | 1 |
| Total per forward pass | 16 |

**Current status**: **37ms avg** after [8,1] herd optimization (was 59ms).

---

## Current AIR Implementation

**Source**: `programming_examples/llama3/swiglu_activation.py` + `swiglu_activation.cc`

| Parameter | Value |
|-----------|-------|
| Herd | [1, 2] (2 cores) |
| Tile size | 1024 elements |
| Vector width | 8 BF16 (in C++ kernel) |
| Memory path | DDR → L1 direct (no L2) |
| Kernel type | External C++ (`swiglu_bf16` in `swiglu_activation.cc`) |
| Compilation | Non-direct-codegen (uses `.o` file) |

### C++ Kernel (`swiglu_activation.cc`)

```cpp
void swiglu_bf16(bfloat16 *gate, bfloat16 *up, bfloat16 *out, int32_t n) {
    constexpr int VecLen = 8;  // 8-wide BF16 vectors
    for (int i = 0; i < n; i += VecLen) {
        g = load_v<8>(gate + i);
        u = load_v<8>(up + i);
        g_half = mul(g, 0.5);
        tanh_val = tanh(g_half);           // Hardware tanh intrinsic
        sigmoid = (1 + tanh_val) * 0.5;
        silu = g * sigmoid;
        result = silu * u;
        store_v(out + i, result);
    }
}
```

### Performance Issues

1. **Only 2 cores** — herd [1,2] uses 2 AIE tiles for 16.7M elements
2. **8-wide vectors** — BF16 on AIE2P supports 16-wide, so 2× underutilized
3. **1024-element tiles** — small tiles mean more DMA overhead per tile
4. **No loop pipelining hints** — missing `AIE_PREPARE_FOR_PIPELINING`

---

## IRON Implementation

**Source**: IRON separates the SwiGLU FFN into 4 independent operators, NOT fused:

| Step | Operator | IRON Kernel | Cores |
|------|----------|------------|-------|
| Gate GEMM | AIEGEMM | `matmul_bf16_bf16` | 32 (8×4) |
| **SiLU** | **AIESiLU** | `silu_tanh_approx_bf16` | **16** (8 cols × 2 ch) |
| **Mul** | **AIEElementwiseMul** | `eltwise_vmul` | **8** (8 cols × 1) |
| Down GEMM | AIEGEMM | `matmul_bf16_bf16` | 32 (8×4) |

### IRON SiLU Kernel

- **Vector width**: 16 BF16 (2× wider than ours)
- **Parallelism**: 16 cores (8× more than our 2)
- **Loop pipelining**: `AIE_PREPARE_FOR_PIPELINING` + `AIE_LOOP_MIN_ITERATION_COUNT(64)`
- **Algorithm**: Same tanh approximation as ours
- **Tile size**: `hidden_dim / 8` = 1024 elements (same as ours)

### IRON ElementwiseMul Kernel

- **Vector width**: 16 BF16
- **Parallelism**: 8 cores
- **Simple**: `load A, load B, mul, store`

### IRON Performance

| Component | IRON standalone (µs) | IRON model (ms) |
|-----------|---------------------|-----------------|
| SwiGLU fused (all 5 ops) | 48,100 | 57.4 |
| SiLU only | — | — |
| Mul only | — | — |

IRON doesn't benchmark SiLU and Mul separately — they're always measured as part of the fused SwiGLU pipeline.

---

## Performance Comparison

| Metric | AIR SwiGLU (step 13 only) | IRON SwiGLU fused (steps 11-14) |
|--------|--------------------------|--------------------------------|
| Latency | **59ms** | **48.1ms** (standalone), 57.4ms (model) |
| What's included | SiLU + mul only | Gate GEMM + Up GEMM + SiLU + mul + Down GEMM |
| Cores | 2 | 16 (SiLU) + 8 (mul) + 32 (GEMMs) |

Our SiLU+mul kernel alone (59ms) takes **longer than IRON's entire FFN block** (48.1ms for 5 ops). This is the single biggest optimization opportunity.

### AIR FFN Block Total (steps 11-14)

| Step | Kernel | Avg (ms) |
|------|--------|---------|
| 11 | GEMM Gate | 24 |
| 12 | GEMM Up | 24 |
| **13** | **SwiGLU (SiLU+mul)** | **59** |
| 14 | GEMM Down | 27 |
| | **Total** | **134** |

Our total FFN: 134ms vs IRON's 48.1ms = **2.8× slower**.

---

## Optimization Opportunities

### 1. Increase parallelism — use [8,1] herd (8 cores)

Same pattern as eltwise_add optimization. Change `num_tiles=2` → herd [8,1] to use 8 AIE columns.

**Expected speedup**: ~4× (2 → 8 cores)
**Complexity**: Low (same pattern as eltwise_add)

### 2. Increase vector width — 16-wide BF16

The C++ kernel uses `VecLen=8` but AIE2P supports 16-wide BF16 vectors. Doubling the vector width doubles throughput.

**Expected speedup**: ~2× (8 → 16 elements per cycle)
**Complexity**: Low (change `constexpr int VecLen = 16`)

### 3. Add loop pipelining hints

IRON's SiLU kernel uses `AIE_PREPARE_FOR_PIPELINING` + `AIE_LOOP_MIN_ITERATION_COUNT(64)` for better instruction scheduling.

**Expected speedup**: ~1.2-1.5×
**Complexity**: Low (add 2 lines to C++ kernel)

### 4. Increase tile size

Current tile_n=1024. Larger tiles reduce DMA overhead. With 16.7M elements and 8 cores, each core processes 2M elements — tile_n=2048 or 4096 would reduce DMA round-trips.

**Expected speedup**: ~1.1-1.2×
**Complexity**: Low

### Optimization Results (2026-03-20)

**16-wide vectors only** (VecLen 8→16, pipelining hints, [1,2] herd): **No improvement** — 59ms → 61ms. Kernel is DMA-bound.

**[8,1] herd + tile_n=4096**: **37ms** (1.6× speedup). The key was increasing tile_n from 1024 to 4096 to keep iterations under the BD limit.

| Config | tile_n | Herd | Iters | BDs (est) | Latency | Status |
|--------|--------|------|-------|----------|---------|--------|
| Old | 1024 | [1,2] | 8192 | 24,576 | 59ms | OK (2 tiles) |
| 16-wide only | 1024 | [1,2] | 8192 | 24,576 | 61ms | OK (no speedup) |
| [8,1] t1024 | 1024 | [8,1] | 2048 | 6,144 | — | **FAIL (BD exhaustion)** |
| **[8,1] t4096** | **4096** | **[8,1]** | **512** | **1,536** | **37ms** | **OK** |

### Root Cause: AIR DMA BD Exhaustion for Large Buffers

The AIR compilation pipeline unrolls all DMA iterations into **separate BD chains**. For 16.7M elements with tile_n=2048:
- Iterations per tile: `16.7M / 2048 = 8192`
- BDs per iteration: 3 (read gate, read up, write output)
- Total BDs per tile: **24,576** — far exceeds hardware limit (~48 BDs per MemTile)

This is NOT compute-bound — with 2 cores the compute takes ~10ms, but DMA transfer of 100MB takes ~50ms.

### BD Limit Threshold (3 buffers, [8,1] herd, tile_n=2048)

| Buffer size | Iterations | BDs (est) | Status |
|------------|-----------|----------|--------|
| 4.2M (eltwise_add) | 256 | 768 | **OK** |
| 8.4M | 512 | 1,536 | **OK** |
| 12.6M | 768 | 2,304 | **OK** |
| 15.7M | 960 | 2,880 | **OK** (max safe) |
| **16.8M (SwiGLU)** | **1024** | **3,072** | **FAIL** |

The BD limit is ~3,072 total (~1,024 iterations × 3 buffers). **Workaround**: increase tile_n to reduce iterations. With tile_n=4096: iterations=512, BDs~1,536 → fits.

### Why IRON Doesn't Have This Problem

IRON uses **ObjectFIFO** for DMA management, which generates **repeating DMA patterns** with fixed BD count (typically 2 for double-buffering). The DMA hardware loops over the same BDs with address updates, regardless of iteration count.

AIR's `dma_memcpy_nd` generates per-iteration BD chains that scale linearly with `n / tile_n`. This is a fundamental architectural limitation of the AIR DMA lowering — it needs the equivalent of ObjectFIFO's BD reuse.

### Remaining Opportunities

| Action | Expected Impact | Blocker |
|--------|----------------|---------|
| Fix AIR BD generation for large buffers | ~4× (enable multi-column herd) | Upstream AIR/aircc DMA lowering |
| Split SwiGLU into 2× 8M chunks | ~2× (each chunk fits [8,1]) | Pipeline modification |
| FFN kernel fusion | ~2× on total FFN block | Complex, requires fused kernel design |

### 5. FFN kernel fusion (future)

Fuse Gate GEMM + Up GEMM + SiLU + mul + Down GEMM into a single dispatch (like IRON). This eliminates 4 DDR round-trips and per-dispatch BO overhead.

**Expected speedup**: ~2× on total FFN block
**Complexity**: High

---

## Reproducible Commands

```bash
cd programming_examples/llama3

# Current performance in LLAMA pipeline
cd build_peano && python3 ../llama3_prefill.py --run-only --n-layers 1 --profile

# Standalone SwiGLU test
make run-swiglu PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

---

## Related Documents

- `performance_optimization.md` — Overall LLAMA optimization roadmap
- `kernels/eltwise_add.md` — Similar optimization pattern (vectorize + [8,1] herd)
- `kernels/gemm.md` — GEMM optimization (completed)
