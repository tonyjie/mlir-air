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

**AIE2P has native `uint4` type**: The `aie::unpack` intrinsic converts uint4 → bfloat16
efficiently in hardware. The kernel loads Q4 packed data to L1 (4x smaller),
unpacks + dequants in-register, then feeds into BF16 MAC.

**Design**:
1. `kernel_builder/mv_q4.cc` — fused Q4 GEMV kernel with signature:
   `q4_matvec_bf16(m, k, row_offset, a_q4, scales, mins, b_in, c_out)`
2. `kernel_builder/quantize.py` — add `pack_q4_for_npu()` packing utility
3. New multi-launch builders for Q4 variant decode kernels
4. `--quantize q4-npu` flag for on-NPU dequant path

**BO layout**: Each weight BO stores `[packed_q4 | scales | mins]` concatenated.
DMA transfers the appropriate slices to L1.

**Challenges**:
- 4-bit unpacking in AIE inner loop (uint8 → 2×uint4 → bfloat16)
- L1 layout: Q4 weight + scale/min metadata DMA patterns
- Function signature change: 8→6 args → multi-launch arg mapping complexity
- Separate `mv_q4.o` / `mv_q4_k8192.o` object files

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
