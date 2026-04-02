# LLAMA-3.2-1B Decode — Progress

## Current Status: First Working Decode Pipeline

**Single-token autoregressive generation working end-to-end on NPU.**

```
Input:  "The capital of France is"
Output: "the capital of France is Paris"
Time:   ~500ms/token (steady state)
IRON:   ~370ms/token
```

### What's Working
- CPU prefill → KV cache populated → NPU decode tokens
- 8 decode kernels compiled: 4 GEMV shapes + rmsnorm + add + rope_q + rope_k
- Per-token decode: 15 NPU kernel invocations + CPU attention + CPU SiLU×mul
- Pre-transposed weights (one-time init, not per-token)
- Correct text generation (produces "Paris" for capital-of-France prompt)

### Performance Breakdown (~500ms/token)

| Component | Estimated time | Notes |
|---|---|---|
| GEMV projections (Q/K/V/O) | ~40ms | 4 × Python invoker overhead (~2ms each) + kernel |
| GEMV FFN (gate/up/down) | ~60ms | 3 × large GEMV + invoker overhead |
| CPU attention | ~5ms | Q @ K_cache.T → softmax → @ V_cache |
| CPU SiLU×mul | <1ms | Tiny (8192 elements) |
| RMSNorm × 2 | ~10ms | Python invoker overhead dominates |
| Eltwise Add × 2 | ~10ms | Python invoker overhead dominates |
| RoPE × 2 | ~10ms | Python invoker overhead dominates |
| Python overhead | ~350ms | Per-layer: weight access, reshaping, bo.write/read |

**Bottleneck**: Python `invoker()` overhead (~2ms per call × ~15 calls/block × 16 blocks = ~480ms). The NPU kernel time is only ~60-80ms based on C++ harness measurements.

---

## GEMV Kernel Status

Detailed analysis: `docs/kernels/gemv.md`

### Optimal Configs (8-column, C++ harness profiling)

| Shape (M×K) | AIR (µs) | IRON (µs) | Gap |
|---|---|---|---|
| 2048 × 2048 (Q/O proj) | 233 | 214 | 1.1x |
| 512 × 2048 (K/V proj) | 81 | 98 | **0.8x (AIR faster)** |
| 8192 × 2048 (FFN gate/up) | 837 | 657 | 1.3x |
| 2048 × 8192 (FFN down) | 946 | 660 | 1.4x |

Best flags: `omit_pingpong=''` (ON), `lock_fix=False`, `tile_sizes=[16,16]`.

### Estimated NPU Kernel Time Per Token

| Kernel | AIR (µs) | Calls/token | Total (ms) |
|---|---|---|---|
| GEMV Q/O | 233 | 32 | 7.5 |
| GEMV K/V | 81 | 32 | 2.6 |
| GEMV gate/up | 837 | 32 | 26.8 |
| GEMV down | 946 | 16 | 15.1 |
| Elementwise (rms+add+rope) | ~100 | ~96 | 9.6 |
| **Total NPU kernel** | | | **~62ms** |

vs IRON standalone: 132ms. **AIR NPU kernels are 2x faster than IRON** (because IRON's SwiGLU Decode fused op at 76ms dominates their total; our decomposed GEMVs are more efficient).

---

## Key Findings

1. **Broadcast DMA bug fixed** — multi-column GEMV (herd_m=8) now works after mlir-air rebuild
2. **Weight transpose critical** — GEMV expects A[M,K], weights stored as (K,M). Must pre-transpose.
3. **Python invoker overhead dominates** — ~2ms per call vs ~100-900µs kernel time. Need C++ harness or bo.map optimization for decode.
4. **L2 staging overhead** — AIR's L2 path adds ~1.3-1.4x vs IRON's direct L3→L1. Needs compiler-level ObjectFIFO-like BD patterns to close.
5. **`runtime_loop_tiling_sizes=[16,16]`** — largest single perf impact (26% improvement over [4,4]).

---

## Files

| File | Purpose |
|---|---|
| `llama3_decode.py` | Main decode pipeline (compile + prefill + decode loop) |
| `docs/kernels/gemv.md` | GEMV kernel analysis (configs, flags, L2 bypass, comparison) |
| `docs/decode/DECODE_PROGRESS.md` | This file |
| `docs/decode/DECODE_PLAN.md` | Original decode plan |
| `docs/decode/iron_decode_reference.md` | IRON baseline numbers |
| `docs/decode/gemv_investigation.md` | Early GEMV investigation notes |
| `decode_kernels/matvec_no_l2.py` | L2 bypass experiment (slower, kept for reference) |
| `decode_kernels/gemv_multi_col.py` | Multi-column herd attempt (pre-broadcast-fix) |

---

## Next Steps

| Priority | Action | Expected improvement |
|---|---|---|
| 1 | **Reduce Python invoker overhead** — use bo.map, static weight BOs, or C++ decode harness | ~500ms → ~100-150ms/token |
| 2 | **Multi-launch merge** — combine GEMV + elementwise ops into fewer ELFs | Reduce dispatch count |
| 3 | **NPU LM Head for decode** — GEMV at (128256, 2048) | Replace CPU LM Head |
| 4 | **NPU SiLU×mul for decode** — move from CPU to NPU | Minor (CPU is fast for 8K elements) |
| 5 | **NPU prefill for KV cache** — use NPU prefill instead of CPU | Faster init |
