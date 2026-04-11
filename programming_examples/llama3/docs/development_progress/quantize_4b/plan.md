# Q4_1 Weight Quantization for LLAMA-3.2-1B on NPU2

## Motivation

Apple-to-apple comparison with FastFlowLM (FLM), the SOTA NPU LLM inference framework.
FLM uses Q4_1 (4-bit asymmetric) quantization on the same NPU2 hardware with Llama-3.2-1B.
Our flow currently runs BF16 end-to-end. Adding Q4 support lets us:

1. Compare accuracy: BF16 vs Q4_1 (how much quality do we lose?)
2. Compare performance: our Q4 vs FLM's Q4 (fair bandwidth comparison)
3. Reduce weight memory: 2.4GB → 0.6GB (4x reduction)

**FLM reference**: [FastFlowLM/Llama-3.2-1B-NPU2](https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2)
- Uses `model.q4nx` format (1.24GB, custom Q4 format)
- On-NPU dequant via `dequant.xclbin` before GEMM/attention
- All compute in BF16 after dequantization

## Q4_1 Format

Per block of 32 contiguous weight values (along last axis):
```
scale = (max - min) / 15          (BF16, one per block)
q = clamp(round((val - min) / scale), 0, 15)  (4-bit unsigned, 32 per block)
dequant = min + q * scale         (BF16, reconstructed)
```

Block size = 32 matches llama.cpp Q4_1 and likely FLM. Asymmetric (scale + min), not
symmetric. Only linear projection weights are quantized — norms stay BF16.

## Phased Approach

### Phase 1: Host-Side Dequant (Accuracy Baseline) — DONE

Quantize weights to Q4_1 on CPU, dequantize back to BF16, load into NPU as BF16.
Same compiled kernels, no NPU changes. Isolates **accuracy impact only**.

**Implementation**: `kernel_builder/quantize.py`
- `quantize_dequant_q4_1(weight, block_size=32)` — per-array Q4→BF16
- `quantize_all_weights(weights, config)` — applies to all linear weights, reports MSE

**Integration**: `--quantize q4` flag in `llama3_inference.py` + `QUANTIZE=q4` in Makefile

**Results**:
| Metric | Value |
|--------|-------|
| Params quantized | 1.24B (7 per layer × 16 layers + embed/lm_head) |
| Size reduction | BF16 2357MB → Q4 589MB (75% smaller) |
| Per-layer MSE | ~2.5e-6 |
| Output quality | All test prompts correct (base + instruct) |
| Performance | Identical to BF16 (NPU still processes BF16) |

Test prompts verified with Q4:
```
Q: What is the capital of France?  → A: Paris ✓
Q: Tell me a joke                  → A: (Pavlov/Schrödinger joke) ✓
Q: 3 apples - 2 = ?               → A: 1 apple ✓
Q: First US president?             → A: George Washington ✓
```

**Usage**:
```bash
make run QUANTIZE=q4
make run MODEL=instruct PROMPT="Tell me a joke" QUANTIZE=q4
```

### Phase 2: Fused Q4 GEMV Kernel (Bandwidth-Fair Comparison) — IN PROGRESS

Fuse Q4 dequantization INTO the GEMV kernel (`mv_q4.cc`), so Q4 weights stay
compressed from DDR all the way to L1. This gives a real 4x DDR bandwidth reduction.

**Why fused, not separate dequant launch**: A separate dequant launch would write
BF16 back to DDR, then GEMV reads it from DDR — same total bandwidth as BF16.
Only fusing dequant into the GEMV inner loop avoids the DDR round-trip.

**Why decode first**: Decode GEMV is bandwidth-bound (92ms/token). Q4 = 4x less
weight data from DDR. Prefill GEMM is compute-bound — less bandwidth benefit.

#### Fused Kernel Approach (attempted)

Created `kernel_builder/mv_q4.cc` — fused Q4 dequant + GEMV in one kernel.
Interleaved weight format: each block = `[16B packed | 2B scale | 2B min]` = 20B.
Single weight buffer (same DMA channel count as BF16). Validated all LLAMA GEMV shapes
on NPU hardware (PASS).

**Performance results** (M=2048, K=2048, [8,1] herd):

| Version | Compile | Correct | Time | vs BF16 |
|---------|---------|---------|------|---------|
| BF16 GEMV (baseline) | OK | PASS | 2.9ms | 1.0x |
| Q4 v1: scalar dequant (`a_vec[i]=`) | OK | PASS | 97ms | 33x slower |
| Q4 v2: scalar to temp buf + vec load | OK | PASS | 95ms | 32x slower |
| Q4 v3: 16-entry LUT per block | OK | PASS | 41ms | 14x slower |
| Q4 v4: vector `bit_and` + `downshift` | **Crash** | — | — | Compiler backend crash |
| Q4 v5: native `uint4` load + `unpack` | OK | **FAIL** | — | Nibble ordering mismatch |

**Root cause**: Scalar dequantization (byte-by-byte nibble extraction + float conversion)
is ~14-33x more instructions than vectorized BF16 MAC. The 3.2x DDR bandwidth
reduction is overwhelmed by compute overhead.

**Vectorization blockers**:
- `aie::bit_and()` on `vector<uint8, 16>` crashes the Peano compiler backend
  (`unable to legalize instruction: G_AND <4 x s32>`)
- Native `uint4` load via `aie::load_v<32>(uint4*)` compiles but produces wrong
  nibble ordering — needs deeper investigation
- No direct `uint4/uint8 × bfloat16` mixed-precision MAC in AIE2P hardware

#### FastFlowLM Architecture (for comparison)

FLM uses a **separate dequant kernel**, not a fused approach:
```
Q4 weights (DDR) → dequant.xclbin (NPU) → BF16 in L2/memtile → mm.xclbin (NPU)
```

Key differences from our fused approach:
- Dequant and GEMM are **separate NPU invocations** with L2 as staging buffer
- Q4→BF16 conversion happens in a dedicated kernel optimized purely for unpacking
- BF16 data stays **on-chip (L2)** between dequant and GEMM — no DDR round-trip
- The actual dequant kernel implementation is proprietary (in `.so` / `.xclbin`)

FLM format: `weight_bf16 = scale * (q_value - zero_point)` with `buffer<u32>` packed
weights, `buffer<bf16>` scales, `buffer<i32>` zero-points. Block size = 32.

#### Potential Next Steps

1. **Two-launch approach** (matching FLM): dequant launch → L2 buffer → GEMV launch.
   Challenge: `air.launch` writes results back to DDR by default. Would need L2-only
   intermediate (possibly via `air.segment`-level buffering or shared L2 allocation).

2. **Fix uint4 nibble ordering**: Debug the native `uint4` load to get correct element
   order. If fixed, the vectorized unpack chain (`uint4→uint8→int16→bfloat16`) could
   be fast enough.

3. **Compiler bug report**: File a bug for the `G_AND` crash on `vector<uint8>` in the
   Peano/LLVM-AIE backend. Once fixed, vector bitwise dequant would work.

4. **Use IRON's dequant**: If IRON (mlir-aie) has a working Q4 dequant kernel, we could
   use it directly (like FLM does).

**Performance target**: Match or approach FLM's decode throughput on the same model/hardware.

### Phase 3: Performance Comparison Report — FUTURE

Side-by-side benchmarks:
- Our BF16 vs our Q4 (host-dequant) vs our Q4 (NPU-dequant) vs FLM Q4
- Prefill and decode latency breakdown
- Memory footprint comparison

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `kernel_builder/quantize.py` | 1 | Q4_1 quantize/dequant utilities |
| `llama3_inference.py` | 1 | `--quantize bf16\|q4` flag |
| `Makefile` | 1 | `QUANTIZE=bf16\|q4` variable |
| `kernel_builder/dequant_q4_bf16.cc` | 2 | NPU dequant kernel (planned) |
| `multi_launch_builder/*` | 2 | Insert dequant launches (planned) |
