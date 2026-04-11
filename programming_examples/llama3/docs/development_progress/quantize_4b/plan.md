# Q4_1 Weight Quantization for LLAMA-3.2-1B on NPU2

## Motivation

Apple-to-apple comparison with FastFlowLM (FLM), the SOTA NPU LLM inference framework.
FLM uses Q4_1 (4-bit asymmetric) quantization on the same NPU2 hardware with Llama-3.2-1B.
Our flow currently runs BF16 end-to-end. Adding Q4 support lets us:

1. Compare accuracy: BF16 vs Q4_1 (how much quality do we lose?)
2. Compare performance: our Q4 vs FLM's Q4 (fair bandwidth comparison)
3. Reduce weight memory: 2.4GB → 0.6GB (4x reduction)

**FLM reference**: [FastFlowLM/Llama-3.2-1B-NPU2](https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2)

## Q4_1 Format

Per block of 32 contiguous weight values (along last axis):
```
scale = (max - min) / 15          (BF16, one per block)
q = clamp(round((val - min) / scale), 0, 15)  (4-bit unsigned, 32 per block)
dequant = min + q * scale         (BF16, reconstructed)
```

Block size = 32 matches llama.cpp Q4_1 and FLM.

---

## Phase 1: Host-Side Dequant (Accuracy Baseline) — DONE

Quantize weights to Q4_1 on CPU, dequantize back to BF16, load into NPU as BF16.
Same compiled kernels, no NPU changes.

**Implementation**: `kernel_builder/quantize.py`
- `quantize_dequant_q4_1()` — per-array Q4→BF16 roundtrip
- `quantize_all_weights()` — applies to all linear weights, reports per-layer MSE

**Integration**: `--quantize q4` in `llama3_inference.py`, `QUANTIZE=q4` in Makefile

**Results**:
| Metric | Value |
|--------|-------|
| Params quantized | 1.24B (7 per layer × 16 + embed/lm_head) |
| Size | BF16 2357MB → Q4 589MB (75% smaller) |
| Per-layer MSE | ~2.5e-6 |
| Output quality | All prompts correct (base + instruct) |
| Performance | Identical to BF16 (NPU still sees BF16) |

**Usage**:
```bash
make run QUANTIZE=q4
make run MODEL=instruct PROMPT="What is the capital of France?" QUANTIZE=q4
```

---

## Phase 2: On-NPU Fused Q4 GEMV Kernel — IN PROGRESS

### Approach: Fused Dequant + GEMV

Created `kernel_builder/mv_q4.cc` — single C++ kernel that reads Q4 packed weights
from L1, dequantizes inline, and computes GEMV. Avoids separate dequant launch.

**Interleaved format**: Each block = `[16B packed Q4 | 2B scale | 2B min]` = 20 bytes.
Single weight buffer (same DMA channel count as BF16 GEMV).

**Validated on NPU** at all LLAMA GEMV shapes: M=2048/8192 K=2048/8192, all PASS.

### Performance History

| Version | M=2048 K=2048 | M=8192 K=2048 | Approach |
|---------|---------------|---------------|----------|
| BF16 baseline | 2.9ms | 19.9ms | Vector MAC, no dequant |
| Q4 v1: scalar dequant | 97ms (33x) | — | `a_vec[i] = bf16(min + q*scale)` per element |
| Q4 v2: temp buffer | 95ms (32x) | — | Write to L1 buf, then vector load |
| Q4 v3: 16-entry LUT | 41ms (14x) | — | Per-block LUT, avoid per-element float math |
| **Q4 v4: reformulated math** | **7.6ms (2.6x)** | **30.2ms (1.5x)** | `min*sum(b) + scale*dot(q,b)` |

### Reformulated Math (Current Best)

```
sum(dequant(q_i) * b_i) = sum((min + q_i * scale) * b_i)
                         = min * sum(b_i) + scale * sum(q_i * b_i)
```

Per block: convert q nibbles to bf16 via global LUT (0-15), then two vector MACs
(`dot(q,b)` and `sum(b)` via `mul(b, ones)`), plus 2 scalar FMAs. Reduces per-block
overhead from 32 float dequants to 32 LUT lookups + 2 scalar ops.

### Vectorization Investigation

Attempted to eliminate the scalar LUT loop with vector intrinsics:

| Approach | Result | Issue |
|----------|--------|-------|
| `aie::bit_and(vec<uint8>, scalar)` | **Compiler crash** | `G_AND <4 x s32>` legalization failure |
| `aie::downshift/upshift` on `vec<uint8>` | **Wrong results** | Shift produces incorrect values |
| `aie::load_v<32>(uint4*)` + `unpack` | **Wrong results** | Nibble ordering mismatch |
| `reduce_add` on `cast_to<float>()` | **Compiler crash** | `G_FADD <16 x s32>` legalization failure |

**Root cause**: Peano/LLVM-AIE compiler has bugs with vector operations on uint8/uint4
types. The bitwise, shift, and type conversion intrinsics exist in the API but don't
lower correctly for these types. This blocks full vectorization of nibble extraction.

### Analysis: Why Q4 Is Still Slower

```
BF16 GEMV (2.9ms):
  Load 64 bf16 → vector MAC → accumulate
  Pure vector ops, no dequant overhead

Q4 GEMV (7.6ms):
  For each block of 32 values:
    16 scalar byte reads + 32 scalar LUT lookups → q_buf[32]     (4.7ms = 61%)
    2 vector loads (q_buf, b) → 2 vector MACs                    (2.9ms = 39%)
    2 scalar reduce_add + 2 scalar FMAs
```

The 61% dequant fraction is entirely scalar byte-by-byte processing. Vectorizing this
(if compiler bugs are fixed) would reduce dequant to ~2-3 vector ops per block,
potentially making Q4 faster than BF16 due to 3.2x less DDR bandwidth.

### FastFlowLM Architecture (for Reference)

FLM uses a fundamentally different approach: **separate dequant + GEMM kernels**:
```
Q4 weights (DDR) → dequant.xclbin (NPU) → BF16 in L2 → mm.xclbin (NPU)
```

Key differences:
- Dequant and GEMM are separate NPU invocations sharing L2 as staging
- BF16 intermediate stays on-chip (L2/memtile), not DDR
- Dequant kernel can be independently optimized for pure unpacking
- Actual implementation is proprietary (in `.so` / `.xclbin`)

**Challenge for our flow**: `air.launch` writes results back to DDR by default.
Implementing L2-only staging between dequant and GEMV launches requires
either segment-level buffer sharing or new MLIR-AIR infrastructure.

### Potential Next Steps

1. **Compiler bug fixes**: Report `bit_and`/`downshift` bugs to LLVM-AIE team.
   Once fixed, vectorized nibble extraction enables full Q4 speedup.

2. **Two-launch with L2 staging**: Match FLM's architecture by sharing L2
   buffer between dequant and GEMV launches within one segment.

3. **Unpacked Q4 format**: Store each Q4 value as uint8 (2x reduction instead
   of 3.2x). Avoids nibble extraction entirely — just `load_v<32>(uint8*)`
   → `unpack` → `to_float` → MAC. Trades compression ratio for speed.

4. **Integrate current kernel**: Even at 2.6x slower (M=2048) or 1.5x slower
   (M=8192), the reformulated kernel is usable for comparison studies.

---

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `kernel_builder/quantize.py` | 1+2 | Q4 quantize/dequant + packing utilities |
| `kernel_builder/mv_q4.cc` | 2 | Fused Q4 GEMV kernel (reformulated math) |
| `matrix_vector_multiplication/q4/matvec_q4.py` | 2 | Standalone Q4 GEMV MLIR builder |
| `test/bench_q4_vs_bf16.py` | 2 | Benchmark script (Q4 vs BF16) |
| `llama3_inference.py` | 1 | `--quantize bf16\|q4` flag |
| `Makefile` | 1 | `QUANTIZE=bf16\|q4` variable |
