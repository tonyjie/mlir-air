# SwiGLU Activation Kernel — Performance Analysis

## Role in LLAMA Pipeline

Step 13 of each transformer block: `output = SiLU(gate) × up`

Where `SiLU(x) = x × sigmoid(x) = x × 0.5 × (tanh(x/2) + 1)`

| Parameter | Value |
|-----------|-------|
| Input size | seq_len × hidden_dim = 2048 × 8192 = **16,777,216** BF16 elements |
| Data volume | 3 buffers × 16.7M × 2B = **100 MB** per invocation |
| Inputs | gate (from GEMM step 11), up (from GEMM step 12) |
| Output | SiLU(gate) × up |
| Invocations | 1 per layer × 16 layers = 16 total |

---

## Current Status

| Config | Herd | tile_n | VecLen | Latency | Status |
|--------|------|--------|--------|---------|--------|
| Original | [1,2] | 1024 | 8 | 59ms | Working |
| **Optimized** | **[8,1]** | **4096** | **16** | **37ms** | **Working** |

**37ms avg** (1.6× speedup from original 59ms).

---

## AIR vs IRON

| Aspect | AIR (current) | IRON |
|--------|--------------|------|
| What it computes | SiLU + eltwise_mul (fused) | SiLU and eltwise_mul (two separate kernels) |
| Vector width | 16 BF16 | 16 BF16 |
| Cores (SiLU) | 8 ([8,1] herd) | 16 (8 cols × 2 channels) |
| Cores (mul) | (fused with SiLU) | 8 (8 cols × 1) |
| DMA management | `dma_memcpy_nd` (per-iteration BDs) | ObjectFIFO (repeating BDs) |
| Loop hints | `AIE_PREPARE_FOR_PIPELINING` | `AIE_PREPARE_FOR_PIPELINING` |
| Algorithm | tanh approximation | tanh approximation (same) |

### FFN Block Comparison (steps 11-14)

| Step | Kernel | AIR (ms) | IRON (ms) |
|------|--------|---------|----------|
| 11 | GEMM Gate (2048×2048×8192) | 24 | (fused) |
| 12 | GEMM Up (2048×2048×8192) | 24 | (fused) |
| 13 | SwiGLU (SiLU+mul, 16.7M) | 37 | (fused) |
| 14 | GEMM Down (2048×8192×2048) | 27 | (fused) |
| | **Total** | **112** | **57.4** (fused, standalone: 48.1) |

AIR FFN = 112ms vs IRON fused = 57.4ms (**2.0× gap**). IRON fuses all 5 ops (Gate+Up+SiLU+mul+Down) into a single dispatch with shared L1/L2 buffers.

---

## Optimization Attempts & Results

### Attempt 1: 16-wide Vectors + Pipelining (keep [1,2] herd)

Updated `swiglu_activation.cc`: VecLen 8→16, added `AIE_PREPARE_FOR_PIPELINING`, pointer-increment loop.

**Result**: 59ms → 61ms — **no improvement**. Kernel is **DMA-bound**, not compute-bound. The actual compute (~10ms) is fast; the 100MB data transfer dominates.

### Attempt 2: Scale to More Cores

Tried multiple herd configurations with tile_n=1024:

| Herd | Cores | Compiles? | Error |
|------|-------|-----------|-------|
| [1,2] | 2 | **YES** | — |
| [2,1] | 2 | **FAIL** | BD exhaustion |
| [1,4] | 4 | **FAIL** | BD exhaustion |
| [4,1] | 4 | **FAIL** | BD exhaustion |
| [8,1] | 8 | **FAIL** | BD exhaustion |

**All multi-core configs fail** at tile_n=1024 with BD exhaustion.

### Attempt 3: Larger Tiles to Reduce BD Count

The BD exhaustion is from too many iterations. Increasing tile_n reduces iterations:

| tile_n | Herd | Iterations | BDs (est) | Compiles? | Latency |
|--------|------|-----------|----------|-----------|---------|
| 1024 | [8,1] | 2048 | 6,144 | **FAIL** | — |
| 2048 | [8,1] | 1024 | 3,072 | **FAIL** | — |
| 2048 | [1,2] | 4096 | 12,288 | **FAIL** | — |
| **4096** | **[8,1]** | **512** | **1,536** | **OK** | **37ms** |

**tile_n=4096 with [8,1] works** — 512 iterations stays under the BD limit.

---

## BD Exhaustion — Root Cause

### What is a Buffer Descriptor (BD)?

A BD is a hardware configuration record that tells the DMA engine where to read/write data. Each BD specifies an address, size, and stride pattern. The NPU2 ShimDMA has **16 BD slots per channel**, each supporting up to **32 repeats** (`repeat_count`).

### The Hardware Constraint

Each ShimDMA channel can address at most:

```
16 BD slots × 32 repeats × tile_n elements = max elements per channel
```

With tile_n=2048: `16 × 32 × 2048 = 1,048,576` elements per channel.
With [8,1] herd: each column has one ShimDMA channel per buffer direction.
Per-column data = `total_n / 8 columns`.

| Total n | Per-column | 16×32×2048 capacity | Fits? |
|---------|-----------|---------------------|-------|
| 4.2M | 524K | 1,048K | **YES** |
| 8.4M | 1,048K | 1,048K | **YES** (exact) |
| 16.8M | 2,097K | 1,048K | **NO** (needs 32 BDs) |
| 16.8M (tile_n=4096) | 2,097K | 2,097K | **YES** (exact) |

### What the Generated IR Looks Like

From `npu.air.mlir` (the failing 16.8M case):

```mlir
aie.runtime_sequence @eltwise_add(%arg0, %arg1, %arg2) {
  // Channel 0_0: Input A, Column 0
  // 16 BDs, each with repeat_count=31 (32 repeats), offset increments by 1M
  %0 = aiex.dma_configure_task_for @air_channel_0_0 {
    aie.dma_bd(%arg0, offset=0, size=4096, strides=[32×32768, 2×16384, 4×512, 512×1])
  } {repeat_count = 31}     // BD 0: covers elements [0, 1M)
  %1 = aiex.dma_configure_task_for @air_channel_0_0 {
    aie.dma_bd(%arg0, offset=1048576, size=4096, ...)
  } {repeat_count = 31}     // BD 1: covers elements [1M, 2M)
  ...
  %15 = aiex.dma_configure_task_for @air_channel_0_0 {
    aie.dma_bd(%arg0, offset=15728640, size=4096, ...)
  } {repeat_count = 31}     // BD 15: covers elements [15M, 16M)
  // ^^^ 16 BDs cover 16 × 32 × 2048 = 1,048,576 elements
  // But column 0 needs 2,097,152 elements → needs 32 BDs → FAIL
}
```

Total: 384 BDs across 24 channels (3 buffers × 8 columns). Each channel uses exactly 16 BDs (the hardware limit), but needs 32 for 16.8M elements.

### How IRON Avoids This (upstream fix direction)

IRON's ObjectFIFO generates **2 BDs per buffer** with hardware address auto-increment:

```
BD 0: transfer to L1 buffer A → hardware increments address
BD 1: transfer to L1 buffer B → hardware increments address
(loop back to BD 0 with new address — works for ANY data size)
```

The upstream fix for AIR would be to generate similar repeating BD patterns in the `dma_memcpy_nd` lowering, instead of unrolling every iteration into a separate BD.

### Reproducible Commands

```bash
cd programming_examples/eltwise_add

# WORKS: 4.2M elements (LLAMA eltwise_add size)
make profile N=4194304

# WORKS: 8.4M elements (at the limit)
make profile N=8388608

# FAILS: 16.8M elements (BD exhaustion — needs 32 BDs, only 16 available)
make profile N=16777216

# WORKS: 16.8M with larger tile (halves BD count)
make profile N=16777216 TILE_N=4096
```

---

## Remaining Optimization Opportunities

| Priority | Action | Expected Impact | Blocker |
|----------|--------|----------------|---------|
| **1** | Upstream: Fix AIR BD generation (repeating patterns) | Enable more cores for large buffers | AIR/aircc compiler change |
| **2** | Split 16.7M into 2 × 8.4M invocations | Each fits [8,1] at tile_n=2048 | Pipeline modification |
| **3** | FFN kernel fusion (Gate+Up+SiLU+mul+Down) | ~2× on total FFN block | Complex kernel design |

---

## Reproducible Commands

```bash
cd programming_examples/llama3

# Compile SwiGLU C++ kernel
make compile-external-kernels PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# Run in LLAMA pipeline
cd build_peano
python3 ../llama3_prefill.py --run-only --n-layers 1 --profile --cpu-attn

# Standalone SwiGLU test
make run-swiglu PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```

---

## Related Documents

- `performance_optimization.md` — Overall LLAMA optimization roadmap
- `kernels/eltwise_add.md` — Similar optimization pattern (same BD limit applies)
- `kernels/gemm.md` — GEMM optimization (completed)
