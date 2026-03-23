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

A BD is a hardware configuration record that tells the DMA engine where to read/write data. Each BD specifies an address, size, and transfer pattern. The NPU2 has a **fixed number of BD slots** per DMA engine (~48 per MemTile).

### How AIR generates DMA instructions

AIR's `dma_memcpy_nd` generates **one BD per DMA transfer**. Each loop iteration creates separate BDs:

```python
for i in range(0, 16777216, tile_n * num_tiles):    # Many iterations
    dma_memcpy_nd(l1_gate, l3_gate, offset=i, size=tile_n)   # 1 BD
    dma_memcpy_nd(l1_up,   l3_up,   offset=i, size=tile_n)   # 1 BD
    # kernel call...
    dma_memcpy_nd(l3_out,  l1_out,  offset=i, size=tile_n)   # 1 BD
```

Total BDs = iterations × 3 buffers. For 16.7M elements:
- tile_n=1024, [8,1]: 2048 iterations × 3 = **6,144 BDs → FAIL**
- tile_n=4096, [8,1]: 512 iterations × 3 = **1,536 BDs → OK**

### How IRON avoids this

IRON uses **ObjectFIFO** which generates **repeating DMA patterns**:

```
ObjectFIFO approach (IRON):
  BD 0: transfer to buffer A  ─┐
  BD 1: transfer to buffer B  ─┤ Hardware loops these 2 BDs
                                │ with auto-incrementing address
  Total: 2 BDs per buffer      │ Works for ANY iteration count
```

```
dma_memcpy_nd approach (AIR):
  BD 0: transfer chunk 0
  BD 1: transfer chunk 1       One BD per iteration
  BD 2: transfer chunk 2       Total: N BDs per buffer
  ...                          Exceeds limit for large N
  BD N: transfer chunk N
```

### BD Limit Threshold

Empirically tested with eltwise_add (3 buffers, [8,1] herd, tile_n=2048):

| Buffer size | Iterations | Status |
|------------|-----------|--------|
| 4.2M | 256 | OK |
| 8.4M | 512 | OK |
| 12.6M | 768 | OK |
| 15.7M | 960 | OK (max safe) |
| **16.8M** | **1024** | **FAIL** |

Limit: ~1,024 iterations with 3 buffers at [8,1] herd. This applies to **any kernel** with large buffers, not just SwiGLU.

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
