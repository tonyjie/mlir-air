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

| Version | M=2048 K=2048 | M=8192 K=2048 | M=512 K=2048 | Approach |
|---------|---------------|---------------|--------------|----------|
| BF16 baseline | 2.9ms | 19.1ms | 0.75ms | Vector MAC, no dequant |
| Q4 v1: scalar dequant | 97ms (33x) | — | — | `a_vec[i] = bf16(min + q*scale)` per element |
| Q4 v2: temp buffer | 95ms (32x) | — | — | Write to L1 buf, then vector load |
| Q4 v3: 16-entry LUT | 41ms (14x) | — | — | Per-block LUT, avoid per-element float math |
| **Q4 v4: reformulated math** | **7.6ms (2.6x)** | **30.2ms (1.6x)** | **2.1ms (2.8x)** | `min*sum(b) + scale*dot(q,b)` |
| Q4 v5: unpacked uint8 | 8.2ms (2.8x) | 32.9ms (1.7x) | — | No nibble packing, more DMA data |
| Q4 v6: per-byte scalar | 150ms (52x) | 601ms (31x) | — | Pure scalar, no vector MAC |
| Q4 v7: unrolled -O3 | 10.5ms (3.6x) | — | — | Pragma unroll, worse (cache pressure) |
| Q4 v8: split-phase | Hang | — | — | L1 overflow from full-row q buffer |

### Reformulated Math (Current Best — v4)

```
sum(dequant(q_i) * b_i) = sum((min + q_i * scale) * b_i)
                         = min * sum(b_i) + scale * sum(q_i * b_i)
```

Per block: convert q nibbles to bf16 via global LUT (0-15), then two vector MACs
(`dot(q,b)` and `sum(b)` via `mul(b, ones)`), plus 2 scalar FMAs. Reduces per-block
overhead from 32 float dequants to 32 LUT lookups + 2 scalar ops.

**Per-block cost**: ~0.13ms/block (consistent across K=256..2048). Total = blocks × 0.13ms.
**Overhead breakdown**: 61% dequant (scalar LUT), 39% vector MAC.

### Exhaustive Optimization Attempts

**Vectorization attempts** (all blocked by compiler bugs):

| Approach | Result | Issue |
|----------|--------|-------|
| `aie::bit_and(vec<uint8>, mask_vec)` | **Compiler crash** | `G_AND <4 x s32>` legalization failure |
| `aie::downshift/upshift` on `vec<uint8>` | **Wrong results** | Vector shift produces incorrect values |
| `aie::load_v<32>(uint4*)` + `unpack` | **Wrong results** | Nibble ordering doesn't match packing |
| `aie::sub(packed, upshift(hi,4))` for lo nibbles | **Wrong results** | Same shift bug as downshift |
| `reduce_add` on `cast_to<float>()` | **Compiler crash** | `G_FADD <16 x s32>` legalization failure |
| Unpacked Q4 + `unpack().cast_to<int16>()` | **Wrong results** | `to_float<bf16>` chain produces incorrect values |

**Other optimization attempts**:

| Approach | Result | Issue |
|----------|--------|-------|
| Unpacked format (36B/block, 1.8x BW) | 8.2ms (2.8x) | More DMA data offsets simpler LUT |
| `-O3` compiler flag | 10.5ms (3.6x) | Worse — likely instruction cache pressure |
| `#pragma clang loop unroll_count(4)` | 10.5ms (3.6x) | Unrolling hurts on AIE |
| Per-byte scalar accumulation | 150ms (52x) | No vector MAC, all scalar |
| Precompute sum(b) cache | 7.7ms (correct fails) | DMA timing issue |
| Split dequant/MAC phases | Hang | L1 overflow from full-row buffer |

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

### What Would Enable Q4 to Outperform BF16

The theoretical minimum (bandwidth-limited) is BF16_time / 3.2 = 0.91ms for M=2048.
Current Q4 = 7.6ms. The 6.7ms gap is purely from 32 scalar LUT lookups per block.

**Remaining paths** (all require changes outside our control):

1. **Compiler bug fixes** (highest impact): File bugs for `bit_and`, `downshift`,
   and `to_float` on vector<uint8/uint4>. Once ANY of these is fixed, the
   entire nibble→bf16 pipeline becomes vector ops (~2 instructions per block
   instead of 32 scalar lookups). Estimated speedup: 5-10x on dequant portion,
   bringing Q4 below BF16.

2. **Two-launch with L2 staging** (FLM architecture): Separate dequant kernel
   writes BF16 to L2, GEMV reads from L2. Requires `air.launch` to support
   L2-only intermediate buffers (currently writes back to DDR).

3. ~~Unpacked Q4 format~~: **Tested and ruled out**. 8.2ms vs 7.6ms packed —
   more DMA data offsets simpler conversion. Same scalar LUT bottleneck.

4. **Integrate current kernel** for accuracy comparison: Even at 2.6x slower,
   the reformulated kernel validates Q4 accuracy end-to-end on NPU. Useful
   for FLM comparison studies without requiring Q4 to be faster.

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
