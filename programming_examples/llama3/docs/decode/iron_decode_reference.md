# IRON Decode Profiling Reference — Llama 3.2 1B

Target numbers to match/beat when implementing decode in MLIR-AIR.

Source: `/home/jiajli/apps/IRON/docs/llama_3.2_1b_profile_decode.md`

---

## End-to-End IRON Decode Performance

Config: prompt_len=2048 (prefill), num_tokens=100 (decode), KV cache enabled.

```
Prefill:         2.87 s  (one-time)
Decode:         33.71 s  (100 tokens)
Tokens/second:   2.94
Time/token:    370 ms
```

---

## Decode Dispatch Trace Per Transformer Block

Each block runs ~12 NPU dispatches for a single decode token:

| # | Operation | Shape | IRON Operator | NPU Cols |
|---|-----------|-------|---------------|----------|
| 1 | RMSNorm (pre-attn) | 2048 elems | AIERMSNorm(2048, 1 col, 2 ch, tile=2048) | 1 |
| 2 | Q Projection | 1 x 2048 -> 2048 | AIEGEMV(M=2048, K=2048) | 8 |
| 3 | K Projection | 1 x 2048 -> 512 | AIEGEMV(M=512, K=2048) | 8 |
| 4 | V Projection | 1 x 2048 -> 512 | AIEGEMV(M=512, K=2048) | 8 |
| 5 | RoPE Q | 32 x 64 (angle_rows=1) | AIERope(32, 64, 1, 1 col) | 1 |
| 6 | RoPE K | 8 x 64 (angle_rows=1) | AIERope(8, 64, 1, 1 col) | 1 |
| 7 | Attention | Q(1,32,64) @ KV_cache | CPU (not NPU) | 0 |
| 8 | O Projection | 1 x 2048 -> 2048 | AIEGEMV(M=2048, K=2048) | 8 |
| 9 | Residual Add | 2048 elems | AIEElementwiseAdd(2048, 1 col) | 1 |
| 10 | RMSNorm (pre-FFN) | 2048 elems | AIERMSNorm(2048, 1 col, 2 ch, tile=2048) | 1 |
| 11 | SwiGLU Decode (fused) | emb=2048, hidden=8192 | AIESwiGLUDecode (5 runlist entries) | 8 |
| 12 | Residual Add | 2048 elems | AIEElementwiseAdd(2048, 1 col) | 1 |

After 16 blocks: Final RMSNorm (1 col) + Final GEMV (128256 x 2048, 8 cols).

Total: ~194 NPU dispatches per decode token.

---

## IRON Standalone Kernel Latencies (Decode Shapes)

### GEMV — Critical Path

| Shape (M x K) | Role | Latency (us) | GFLOP/s | Bandwidth (GB/s) | Corr |
|---|---|---|---|---|---|
| 2048 x 2048 | Q/O projection | 214 | 39.2 | 39.3 | 1.00000 |
| 512 x 2048 | K/V projection | 98 | 21.4 | 21.5 | 1.00000 |
| 8192 x 2048 | FFN gate/up | 657 | 51.1 | 51.1 | 1.00000 |
| 2048 x 8192 | FFN down | 660 | 50.8 | 50.9 | 1.00000 |
| 128256 x 2048 | Final vocab | 9,443 | 55.6 | 55.7 | 1.00000 |

Config: `is_mv=False` for projections (tile_in=1), `is_mv=True` for final vocab (tile_in=4).

### Other Decode Kernels

| Kernel | Shape | Latency (us) | Corr | Notes |
|---|---|---|---|---|
| RoPE Q (decode) | 32 x 64, angle_rows=1 | 39 | 1.00000 | 1 col |
| RoPE K (decode) | 8 x 64, angle_rows=1 | 38 | 1.00000 | 1 col |
| RMSNorm (decode) | 2048, weighted | 42 | 0.99999 | 1 col, 2 ch |
| ElementwiseAdd (decode) | 2048 | 44 | 0.99998 | 1 col |
| SwiGLU Decode (fused) | emb=2048, hidden=8192 | 4,766 | 0.99999 | 8 cols, 5 entries |

---

## Per-Token Time Budget

Decode = 370 ms/token. Breakdown:

| Component | Per-token (ms) | % | Source |
|---|---|---|---|
| GQA block (4x GEMV + 2x RoPE + CPU attn + GEMV out) | ~110 | 30% | 16 blocks |
| SwiGLU Decode (fused FFN) | ~112 | 30% | 16 blocks |
| RMSNorm (3x per block + 1 final) | ~37 | 10% | 49 calls |
| ElementwiseAdd (2x per block) | ~28 | 8% | 32 calls |
| RoPE (2x per block) | ~17 | 5% | 32 calls |
| Final norm + final GEMV | ~11 | 3% | 1 call |
| Python/profiler/CPU overhead | ~55 | 15% | — |

### Standalone Kernel Total Per Token

| Kernel | Standalone (us) | Calls/token | Total (ms) |
|---|---|---|---|
| GEMV Q/O (2048x2048) | 214 | 32 | 6.8 |
| GEMV K/V (512x2048) | 98 | 32 | 3.1 |
| GEMV FFN gate/up (8192x2048) | 657 | 32 | 21.0 |
| GEMV FFN down (2048x8192) | 660 | 16 | 10.6 |
| GEMV final vocab (128256x2048) | 9,443 | 1 | 9.4 |
| SwiGLU Decode fused | 4,766 | 16 | 76.3 |
| RoPE decode Q | 39 | 16 | 0.6 |
| RoPE decode K | 38 | 16 | 0.6 |
| RMSNorm decode | 42 | 49 | 2.1 |
| ElementwiseAdd decode | 44 | 32 | 1.4 |
| **Total standalone** | | | **131.9** |

Standalone total (132 ms) vs measured (370 ms) = ~2.8x overhead from Python
`forward()`, buffer sync, CPU attention, and `sys.setprofile()` instrumentation.

---

## Key Differences: Decode vs Prefill

| Aspect | Prefill | Decode |
|---|---|---|
| Sequence length | 2048 tokens | 1 token |
| Matrix ops | GEMM (MxKxN) | GEMV (MxK, vector input) |
| Attention | Fused MHA on NPU (8 pipelines) | CPU-based (KV cache lookup) |
| FFN | SwiGLU Prefill (GEMM-based) | SwiGLU Decode (GEMV-based, fused 5 entries) |
| RMSNorm/Add size | seq_len x emb_dim (4.2M) | emb_dim (2048) |
| RMSNorm/Add cols | 8 AIE columns | 1 AIE column |
| Bottleneck | GEMM compute (TFLOP/s) | GEMV bandwidth + dispatch overhead |

---

## AIR Implementation Priorities

Based on the profiling, the highest-impact kernels for decode are:

1. **GEMV** — 60% of standalone kernel time. 5 distinct shapes needed.
   Priority: FFN gate/up (8192x2048) and FFN down (2048x8192) are largest.

2. **SwiGLU Decode** — Can be decomposed as 2x GEMV + SiLU + Mul + GEMV.
   IRON fuses these into a single xclbin. AIR can start with separate kernels.

3. **Attention** — IRON runs on CPU during decode. If AIR can implement a
   fast single-query attention on NPU, this could be a differentiator.

4. **RMSNorm / Add / RoPE** — Small tensors (2048 elements), very fast on NPU
   (~40 us each). Low priority for optimization but needed for correctness.

5. **Final vocab GEMV** — Single large (128256x2048) dispatch at 9.4 ms.
   Can use is_mv=True with tile_in=4.
