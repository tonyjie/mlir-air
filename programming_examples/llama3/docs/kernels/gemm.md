# GEMM Kernel — Performance & Precision Analysis

## Role in LLAMA Pipeline

7 GEMM invocations per transformer block (112 total across 16 layers):

| Step | Projection | Shape (M×K×N) | Count/layer |
|------|-----------|---------------|-------------|
| 2 | Q projection | 2048 × 2048 × 2048 | 1 |
| 3 | K projection | 2048 × 2048 × 512 | 1 |
| 4 | V projection | 2048 × 2048 × 512 | 1 |
| 8 | O projection | 2048 × 2048 × 2048 | 1 |
| 11 | Gate FFN | 2048 × 2048 × 8192 | 1 |
| 12 | Up FFN | 2048 × 2048 × 8192 | 1 |
| 14 | Down FFN | 2048 × 8192 × 2048 | 1 |

---

## AIE2P Hardware: GEMM Accumulator Behavior

The AIE2P `aie::mmul` intrinsic **always accumulates in F32 internally** (`accauto` → `accfloat`). The precision difference is **when F32→BF16 truncation happens**:

### Two modes controlled by `prio_accuracy` flag

| Mode | Compilation Flag | Internal Accum | Output Buffer | Truncation Point |
|------|-----------------|---------------|---------------|------------------|
| Standard | `-Dbf16_bf16_ONLY` | F32 | **BF16** (2B/elem) | Per K-tile: `to_vector<bfloat16>()` after each tile |
| Priority Accuracy | `-Dbf16_f32_ONLY` | F32 | **F32** (4B/elem) | Deferred: `convert_copy_f32_to_bf16()` after full K reduction |

### Why this matters for precision

With `prio_accuracy=False` (standard), the F32 accumulator is truncated to BF16 after each K-tile (e.g., every 8 elements of K). For large K (like K=8192 in Down GEMM), this means 1024 truncation points, each losing ~0.4% precision. These errors compound across the K reduction.

With `prio_accuracy=True`, F32 is preserved through the entire K-reduction (all 8192 elements), and truncated only once at the end. This gives much higher precision but uses 2× the output buffer memory (F32 vs BF16).

### mmul tile dimensions

| Config | `emulate_bf16_mmul_with_bfp16` | mmul dims (r×s×t) |
|--------|-------------------------------|-------------------|
| BFP16 emulation (default) | True | 8×8×8 |
| Native BF16 | False | 4×8×8 |

BFP16 emulation uses block floating point (shared exponent across a block), giving 8×8×8 tiles but slightly lower precision than native BF16.

### C++ kernel code path

```
design.py: prio_accuracy flag
  → op.py: appends "-Dbf16_bf16_ONLY" or "-Dbf16_f32_ONLY"
  → mm.cc: instantiates matmul_vectorized_<dims>_bf16_<out>()
    → aie::mmul<r,s,t, bfloat16, bfloat16, accauto>  (accauto = accfloat = F32)
    → inner loop: C.mac(A, B)  // F32 accumulation
    → output: C.to_vector<T_out>()  // T_out = bfloat16 or float
```

Source: `/home/jiajli/apps/IRON/aie_kernels/aie2p/mm.cc`

---

## IRON LLAMA Configuration

**All GEMMs use `prio_accuracy=False`** (BF16 output, per-K-tile truncation).

| Projection | Shape | `prio_accuracy` | Output dtype | IRON Latency (µs) |
|-----------|-------|-----------------|-------------|-------------------|
| Q/O | 2048×2048×2048 | False | BF16 | 7,367 |
| K/V | 2048×2048×512 | False | BF16 | 1,650 |
| Gate/Up | 2048×2048×8192 | False | BF16 | (fused in SwiGLU) |
| Down | 2048×8192×2048 | False | BF16 | (fused in SwiGLU) |
| SwiGLU fused (all FFN) | — | False | BF16 | 57,700 (total) |

IRON benchmark commands:
```bash
cd /home/jiajli/apps/IRON
pytest iron/operators/gemm/test.py -k "gemm_2048x2048x2048" -v -s     # Q/O shape
pytest iron/operators/gemm/test.py -k "llama_kv_proj_2048tok" -v -s    # K/V shape
pytest iron/operators/swiglu_prefill/test.py -m "llama and extensive" -v -s  # fused FFN
```

### IRON GEMM design parameters

From `/home/jiajli/apps/IRON/iron/operators/gemm/design.py`:
- Tile size: 64×64×64 (default for 8-col config)
- Columns: 8 (one worker per AIE column)
- `emulate_bf16_mmul_with_bfp16`: True (8×8×8 mmul)
- `round_conv_even`: True
- Output: BF16 buffer (no F32 intermediate)

---

## Our Current Implementation (AIR)

**Source**: `programming_examples/matrix_multiplication/bf16/run.py`

| Parameter | Value |
|-----------|-------|
| Input dtype | BF16 |
| Output dtype | **F32** (`build_gemm(..., np.float32)`) |
| Accumulator | F32 (hardware) |
| Tile config | tile_m=32, tile_k_l2=64, tile_k_l1=32, tile_n=32 |
| Herd | [4, 4] (16 cores) |
| Vectorization | Transform IR (tiling + unrolling + vector type cast) |
| BFP16 emulation | Yes (`-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`) |

### Why we use F32 output

We chose F32 output to preserve precision across the K-reduction. The Down GEMM (K=8192) showed:

| Output dtype | Per-step corr vs CPU F32 | max_err |
|-------------|-------------------------|---------|
| BF16 output | 0.948 | 7.35 |
| **F32 output** | **0.9998** | **0.16** |

The 52× precision improvement from F32 output was critical when we were using F32 residual adds. Now that we've switched to BF16 residual adds (matching IRON), we could re-evaluate whether BF16 GEMM output is sufficient.

---

## AIR vs IRON Comparison

### Data Type Flow

| Stage | IRON | AIR (ours) |
|-------|------|-----------|
| GEMM input A | BF16 | BF16 |
| GEMM input B (weights) | BF16 | BF16 |
| Hardware accumulator | F32 | F32 |
| **GEMM output buffer** | **BF16** | **F32** |
| Cast before next kernel | None (already BF16) | `.astype(bfloat16)` |
| Data volume per output | 2 bytes/elem | 4 bytes/elem |

### Performance Comparison (per-invocation, with XRT context reuse)

| GEMM Shape | AIR (ms) | IRON (ms) | Gap | Notes |
|-----------|---------|-----------|-----|-------|
| Q/O: 2048×2048×2048 | 33 | 7.4 | 4.5× | |
| K/V: 2048×2048×512 | 8 | 1.65 | 4.8× | |
| Gate/Up: 2048×2048×8192 | 111 | (fused) | — | IRON fuses into SwiGLU |
| Down: 2048×8192×2048 | 91 | (fused) | — | |
| FFN total (steps 11-14) | 390 | 57.7 (fused) | 6.8× | |

### Performance Gap Sources

1. **F32 output doubles DMA volume**: Our GEMM outputs F32 (4B/elem), IRON outputs BF16 (2B/elem). For Q/O projection (2048×2048 output), that's 16MB vs 8MB per transfer.

2. **Tile configuration**: Our tiles (32×32) are smaller than IRON's (64×64), potentially less compute-efficient.

3. **Herd shape**: We use [4,4] = 16 cores; IRON uses 8 columns. Our larger herd may cause more DMA contention.

4. **Kernel fusion**: IRON's FFN block runs as a single SwiGLU dispatch. Our 4 separate dispatches (Gate + Up + SwiGLU + Down) each pay per-invocation overhead and lose L1/L2 data locality.

---

## Optimization Opportunities

### 1. Switch GEMM output to BF16

Match IRON's approach: output BF16 instead of F32. This halves DMA volume and eliminates the F32→BF16 cast.

**Risk**: Per-kernel correlation drops (0.9998 → 0.948 for Down GEMM). Need to verify 16-layer model still produces correct output.

**IRON validates this works**: Their pure BF16 pipeline produces correct model output.

### 2. Tile configuration tuning

Try larger tiles (64×64) to match IRON and improve compute-to-DMA ratio.

### 3. FFN kernel fusion

Fuse Gate GEMM + Up GEMM + SwiGLU activation + Down GEMM into a single kernel dispatch. IRON's fused SwiGLU prefill runs at 57.7ms vs our 390ms (6.8× gap).

### 4. Herd shape optimization

Experiment with different herd shapes (e.g., [8,4] vs [4,4]) to find optimal core utilization for each GEMM shape.

---

## Profiling Commands

```bash
# AIR: Profile GEMM standalone
cd programming_examples/matrix_multiplication/bf16
make profile AIE_TARGET=aie2p   # 1024×1024×1024 default

# AIR: Profile in LLAMA pipeline
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 1 --profile

# IRON: Profile GEMM at LLAMA shapes
cd /home/jiajli/apps/IRON
pytest iron/operators/gemm/test.py -k "gemm_2048x2048x2048" -v -s
pytest iron/operators/gemm/test.py -k "llama_kv_proj_2048tok" -v -s
pytest iron/operators/swiglu_prefill/test.py -m "llama and extensive" -v -s
```

---

## Related Documents

- `LLAMA_gemm.md` — GEMM precision investigation (F32 output fix, BFP16 emulation analysis)
- `performance_optimization.md` — Overall LLAMA optimization roadmap
- `kernels/eltwise_add.md` — Eltwise add optimization (completed)
