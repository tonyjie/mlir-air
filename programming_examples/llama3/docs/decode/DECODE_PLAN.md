# LLAMA-3.2-1B Decode on MLIR-AIR (NPU2) -- Plan

## Context

Prefill (seq_len=2048) is near-complete on AIR, matching IRON performance.
This plan covers the decode phase: single-token generation with KV cache.

**IRON baseline**: 2.94 tok/s, 370 ms/token (with Python overhead).
**Standalone kernel total**: 132 ms/token (pure NPU dispatch time).
**Target**: Match or exceed IRON's standalone kernel throughput.

---

## Per-Block Kernel Sequence (Decode, single token)

Each transformer block runs 12 kernel invocations:

| # | Operation | Shape | AIR Kernel Needed | Status |
|---|-----------|-------|-------------------|--------|
| 1 | RMSNorm (pre-attn) | (2048,) + weight(2048,) | weighted_rms_norm (1 col) | TODO |
| 2 | Q Projection | (1, 2048) @ W(2048, 2048) | GEMV | TODO |
| 3 | K Projection | (1, 2048) @ W(2048, 512) | GEMV | TODO |
| 4 | V Projection | (1, 2048) @ W(2048, 512) | GEMV | TODO |
| 5 | RoPE on Q | (32, 64) with single angle | rope (1 col) | TODO |
| 6 | RoPE on K | (8, 64) with single angle | rope (1 col) | TODO |
| 7 | Attention | Q(1,32,64) @ KV_cache | CPU or NPU attention | TODO |
| 8 | O Projection | (1, 2048) @ W(2048, 2048) | GEMV | TODO |
| 9 | Residual Add | (2048,) + (2048,) | eltwise_add (1 col) | TODO |
| 10 | RMSNorm (pre-FFN) | (2048,) + weight(2048,) | weighted_rms_norm (1 col) | TODO |
| 11-15 | SwiGLU FFN | gate/up GEMV + SiLU + Mul + down GEMV | decomposed ops | TODO |
| 16 | Residual Add | (2048,) + (2048,) | eltwise_add (1 col) | TODO |

After 16 blocks: Final RMSNorm + LM Head GEMV (128256 x 2048).

---

## New Kernels Required

### GEMV (matrix-vector multiply)

Not present in prefill pipeline. This is the main new kernel for decode.

| Shape (M x K) | Role | IRON Latency (us) | Calls/token |
|---|---|---|---|
| 2048 x 2048 | Q/O projection | 214 | 32 |
| 512 x 2048 | K/V projection | 98 | 32 |
| 8192 x 2048 | FFN gate/up | 657 | 32 |
| 2048 x 8192 | FFN down | 660 | 16 |
| 128256 x 2048 | Final vocab | 9,443 | 1 |

IRON uses `is_mv=False` (tile_in=1) for projections, `is_mv=True` (tile_in=4)
for final vocab. All use 8 AIE columns.

**Key difference from GEMM**: Input is a single vector (K elements), output is
M elements. Compute is O(M*K) not O(M*N*K). Memory-bandwidth bound, not compute
bound. Peak ~55 GB/s bandwidth on NPU2.

### Existing Kernels (adapt to decode shapes)

These kernels already exist for prefill but need to work at decode sizes:

| Kernel | Prefill shape | Decode shape | Change needed |
|---|---|---|---|
| weighted_rms_norm | (2048*2048,) 8 cols | (2048,) 1 col | Reduce to 1 col |
| eltwise_add | (2048*2048,) 8 cols | (2048,) 1 col | Reduce to 1 col |
| rope_lut | (65536, 64) 8 cols | (32, 64) 1 col | Single token angles |
| silu_and_mul | (16777216,) 8 cols | (8192,) 1 col | Reduce to 1 col |

---

## Implementation Order

### Phase 1: GEMV kernel
1. Implement GEMV kernel for AIE2P (similar to IRON's `mv.cc`)
2. Validate at Llama shapes: 2048x2048, 512x2048, 8192x2048, 2048x8192
3. Benchmark against IRON: target >= 50 GFLOP/s for large shapes

### Phase 2: Decode-size elementwise ops
4. Test RMSNorm at size=2048, 1 col
5. Test ElementwiseAdd at size=2048, 1 col
6. Test RoPE at (32,64) and (8,64), 1 col
7. Test SiLU+Mul at size=8192, 1 col

### Phase 3: SwiGLU Decode (decomposed)
8. Combine: GEMV(gate) + GEMV(up) + SiLU+Mul + GEMV(down)
9. Validate end-to-end FFN decode correctness

### Phase 4: Single decode token pipeline
10. Integrate all kernels into a single-token decode pass
11. Add KV cache management (CPU-side or NPU-side)
12. Implement CPU attention (or NPU single-query attention)

### Phase 5: Multi-token generation
13. Loop decode for N tokens with KV cache accumulation
14. End-to-end timing vs IRON baseline

---

## IRON Reference

Detailed IRON profiling data: `decode/iron_decode_reference.md`
IRON source code: `/home/jiajli/apps/IRON/iron/operators/gemv/`
IRON GEMV kernel: `/home/jiajli/apps/IRON/aie_kernels/generic/mv.cc`
