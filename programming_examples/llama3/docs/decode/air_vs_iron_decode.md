# AIR vs IRON Decode — Step-by-Step Comparison

## Per Transformer Block (×16 blocks per token)

| Step | Operation | IRON | AIR (current) | Same? |
|---|---|---|---|---|
| 1 | **RMSNorm (pre-attn)** | NPU: AIERMSNorm, 1 col, 2 ch, tile=2048 | NPU: weighted_rms_norm, M=1 N=2048, xclbin | **Similar** — both NPU, AIR uses prefill kernel at M=1 |
| 2 | **Q GEMV** | NPU: AIEGEMV, M=2048 K=2048, 8 cols, tile_in=1, tile_out=128 | NPU: matvec, M=2048 K=2048, herd_m=8, tile_m=8, m_input=4 | **Similar** — both 8-col NPU GEMV. Different tiling params. |
| 3 | **K GEMV** | NPU: AIEGEMV, M=512 K=2048, 8 cols | NPU: matvec, M=512 K=2048, herd_m=8 | **Similar** |
| 4 | **V GEMV** | NPU: AIEGEMV, M=512 K=2048, 8 cols | NPU: matvec, M=512 K=2048, herd_m=8 | **Similar** |
| 5 | **RoPE Q** | NPU: AIERope, rows=32 cols=64, angle_rows=1, 1 col | NPU: rope_lut, (32,64), xclbin | **Similar** — both NPU, single position angle |
| 6 | **RoPE K** | NPU: AIERope, rows=8 cols=64, angle_rows=1, 1 col | NPU: rope_lut, (8,64), xclbin | **Similar** |
| 7a | **KV Cache Update** | CPU: write to torch tensor at cache_pos | CPU: write to numpy array at current_pos | **Same** — both CPU-managed DDR cache |
| 7b | **GQA Expansion** | CPU: repeat_interleave(4, dim=1) to expand 8→32 heads | CPU: loop over 32 heads, index kv_h = h // 4 | **Same logic** — IRON expands then batches; AIR loops per-head |
| 7c | **Attention** | CPU: torch.sdpa (Q @ K.T / sqrt(64) → softmax → @ V) | CPU: numpy manual (same math, per-head loop) | **Same** — both CPU, no NPU attention for decode |
| 8 | **O GEMV** | NPU: AIEGEMV, M=2048 K=2048, 8 cols | NPU: matvec, M=2048 K=2048, herd_m=8 | **Similar** |
| 9 | **Residual Add** | NPU: AIEElementwiseAdd, 1 col, 2 ch, tile=2048 | NPU: eltwise_add, n=2048, xclbin | **Similar** |
| 10 | **RMSNorm (pre-FFN)** | NPU: AIERMSNorm, 1 col | NPU: weighted_rms_norm, M=1 N=2048 | **Similar** |
| 11 | **Gate GEMV** | NPU: fused in SwiGLU (AIEGEMV 8192×2048, tile_in=4) | NPU: matvec, M=8192 K=2048, herd_m=8 | **Different** — IRON fused, AIR separate |
| 12 | **Up GEMV** | NPU: fused in SwiGLU | NPU: matvec, M=8192 K=2048, herd_m=8 | **Different** |
| 13 | **SiLU × mul** | NPU: fused in SwiGLU (SiLU + Mul entries) | **CPU**: numpy SiLU × mul (8192 elements) | **Different** — IRON NPU fused, AIR CPU |
| 14 | **Down GEMV** | NPU: fused in SwiGLU (AIEGEMV 2048×8192) | NPU: matvec, M=2048 K=8192, herd_m=8 | **Different** — IRON fused, AIR separate |
| 15 | **Residual Add** | NPU: AIEElementwiseAdd, 1 col | NPU: eltwise_add, n=2048 | **Similar** |

## After 16 Blocks

| Step | IRON | AIR (current) | Same? |
|---|---|---|---|
| Final RMSNorm | NPU: AIERMSNorm | CPU: numpy rms_norm | **Different** — IRON NPU, AIR CPU |
| LM Head GEMV | NPU: AIEGEMV 128256×2048, 8 cols, is_mv=True | CPU: numpy matmul | **Different** — IRON NPU, AIR CPU |
| Token selection | CPU: argmax/sampling | CPU: argmax | **Same** |

---

## Key Differences

### 1. FFN: Fused vs Decomposed

| Aspect | IRON | AIR |
|---|---|---|
| FFN structure | **1 fused SwiGLU op** (5 runlist entries in 1 xclbin) | **3 separate NPU GEMV calls + 1 CPU SiLU×mul** |
| NPU dispatches for FFN | 1 (fused) | 3 (gate + up + down GEMVs) |
| SiLU × mul | NPU (inside fused op) | CPU (numpy) |
| Intermediates | Stay on NPU between entries | Round-trip through DDR between GEMV calls |
| IRON latency | 4766µs per block | ~3× GEMV time + Python overhead |

This is the biggest architectural difference. IRON keeps FFN intermediates on-device; AIR reads them back to host between each GEMV.

### 2. GEMV Kernel Architecture

| Aspect | IRON | AIR |
|---|---|---|
| Data path | DDR → L1 direct (ObjectFIFO) | DDR → L2 → L1 (MemTile staging) |
| BD pattern | 2 BDs per FIFO (auto-increment) | Individual BDs per DMA transfer |
| Weight layout | Row-major, streamed via ObjectFIFO | Row-major, staged through L2 |
| Achieved BW | 21-51 GB/s | 22-40 GB/s |

### 3. Weight Management

| Aspect | IRON | AIR |
|---|---|---|
| Weight storage | Static BOs, pre-loaded at init | Pre-transposed numpy arrays, bo.write per call |
| Weight sync | `bo.sync()` only (no write per call) | `bo.write()` + `bo.sync()` every call |
| Weight transpose | Done at model init | Done at model init (pre-transposed) |

### 4. Buffer Management

| Aspect | IRON | AIR |
|---|---|---|
| Input/output | `bo.map()` zero-copy | Python invoker: `bo.write()` + `bo.read()` per call |
| Overhead per dispatch | ~0.1-0.5ms | ~2ms (Python invoker dominance) |
| Total dispatches/block | ~12 | ~15 |
| Total dispatch overhead/token | ~30ms | ~480ms |

### 5. Final Layers

| Aspect | IRON | AIR |
|---|---|---|
| Final RMSNorm | NPU | CPU |
| LM Head | NPU GEMV (128256×2048, 8 cols) | CPU matmul |

---

## NPU Invocations Per Block

| IRON (~12 dispatches) | AIR (~15 dispatches) |
|---|---|
| 1. RMSNorm | 1. RMSNorm |
| 2. Q GEMV | 2. Q GEMV |
| 3. K GEMV | 3. K GEMV |
| 4. V GEMV | 4. V GEMV |
| 5. RoPE Q | 5. RoPE Q |
| 6. RoPE K | 6. RoPE K |
| 7. (CPU attention) | 7. (CPU attention) |
| 8. O GEMV | 8. O GEMV |
| 9. Residual Add | 9. Residual Add |
| 10. RMSNorm | 10. RMSNorm |
| 11. **SwiGLU fused (1 dispatch, 5 entries)** | 11. Gate GEMV |
| 12. Residual Add | 12. Up GEMV |
| | 13. **(CPU SiLU×mul)** |
| | 14. Down GEMV |
| | 15. Residual Add |

---

## Performance Comparison

| Metric | IRON | AIR | Gap |
|---|---|---|---|
| **Standalone NPU kernel time** | 132ms/token | **~62ms/token** (estimated) | **AIR 2x faster** |
| **End-to-end per token** | 370ms | ~500ms | AIR 1.4x slower |
| **Python/dispatch overhead** | ~55ms | ~430ms | AIR 8x more overhead |
| **Total NPU dispatches/token** | ~194 | ~240+ | AIR 24% more |

**The NPU kernel compute is faster in AIR** (decomposed GEMVs avoid IRON's fused SwiGLU overhead of 4766µs). But the Python invoker overhead (~2ms per call × 240 calls) dominates AIR's end-to-end time.

---

## What Would Close the Gap

| Action | Estimated impact |
|---|---|
| **Static weight BOs** (write once, not per call) | Eliminate ~200ms of bo.write |
| **bo.map() zero-copy** (like prefill) | Eliminate ~100ms of bo.read |
| **Fused SwiGLU** (multi-launch ELF) | Reduce 3 FFN dispatches → 1 |
| **NPU Final RMSNorm + LM Head** | Replace CPU compute |
| **C++ decode harness** | Eliminate Python invoker overhead entirely |
