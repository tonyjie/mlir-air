# AIR vs IRON Decode — Step-by-Step Comparison

## Per Transformer Block (x16 blocks per token)

| Step | Operation | IRON | AIR (current) | Same? |
|---|---|---|---|---|
| 1 | **RMSNorm (pre-attn)** | NPU: AIERMSNorm, 1 col, 2 ch, tile=2048 | NPU: weighted_rms_norm, M=1 N=2048, [1,1] herd | **Similar** |
| 2 | **Q GEMV** | NPU: AIEGEMV, M=2048 K=2048, 8 cols | NPU: matvec, M=2048 K=2048, herd_m=8 (merged in qkv_gemv ELF) | **Similar** |
| 3 | **K GEMV** | NPU: AIEGEMV, M=512 K=2048, 8 cols | NPU: matvec, M=512 K=2048, herd_m=8 (merged in qkv_gemv ELF) | **Similar** |
| 4 | **V GEMV** | NPU: AIEGEMV, M=512 K=2048, 8 cols | NPU: matvec, M=512 K=2048, herd_m=8 (merged in qkv_gemv ELF) | **Similar** |
| 5 | **RoPE Q** | NPU: AIERope, rows=32 cols=64, 1 col | NPU: rope_lut, (32,64), [1,1] herd | **Similar** |
| 6 | **RoPE K** | NPU: AIERope, rows=8 cols=64, 1 col | NPU: rope_lut, (8,64), [1,1] herd | **Similar** |
| 7a | **KV Cache Update** | CPU: write to torch tensor at cache_pos | CPU: write to numpy array at current_pos | **Same** |
| 7b | **GQA Expansion** | CPU: repeat_interleave(4, dim=1) | CPU: loop over 32 heads, kv_h = h // 4 | **Same logic** |
| 7c | **Attention** | CPU: torch.sdpa | CPU: numpy manual (Q @ K.T / sqrt → softmax → @ V) | **Same** |
| 8 | **O GEMV** | NPU: AIEGEMV, M=2048 K=2048, 8 cols | NPU: matvec, herd_m=8 (merged in o_gemv_add ELF) | **Similar** |
| 9 | **Residual Add** | NPU: AIEElementwiseAdd, 1 col | NPU: eltwise_add, [8,1] herd (merged in o_gemv_add ELF) | **Similar** |
| 10 | **RMSNorm (pre-FFN)** | NPU: AIERMSNorm, 1 col | NPU: weighted_rms_norm, M=1 N=2048, [1,1] herd | **Similar** |
| 11 | **Gate GEMV** | NPU: fused in SwiGLU | NPU: matvec, M=8192 K=2048 (merged in gate_up_gemv ELF) | **Different** — IRON fused, AIR merged gate+up only |
| 12 | **Up GEMV** | NPU: fused in SwiGLU | NPU: matvec, M=8192 K=2048 (merged in gate_up_gemv ELF) | **Different** |
| 13 | **SiLU x mul** | NPU: fused in SwiGLU | NPU: silu_and_mul, n=8192, [8,1] herd (separate call) | **Different** — IRON fused, AIR separate NPU kernel |
| 14 | **Down GEMV** | NPU: fused in SwiGLU | NPU: matvec, M=2048 K=8192, herd_m=8 (separate call) | **Different** — IRON fused, AIR separate |
| 15 | **Residual Add** | NPU: AIEElementwiseAdd, 1 col | NPU: eltwise_add, [8,1] herd (separate call) | **Similar** |

## After 16 Blocks

| Step | IRON | AIR (current) | Same? |
|---|---|---|---|
| Final RMSNorm | NPU: AIERMSNorm | CPU: numpy rms_norm | **Different** — IRON NPU, AIR CPU |
| LM Head GEMV | NPU: AIEGEMV 128256x2048, 8 cols | CPU: numpy matmul | **Different** — IRON NPU, AIR CPU |
| Token selection | CPU: argmax/sampling | CPU: argmax | **Same** |

---

## Key Architectural Differences

### 1. FFN: Fused vs Decomposed

| Aspect | IRON | AIR |
|---|---|---|
| FFN structure | **1 fused SwiGLU op** (5 runlist entries in 1 xclbin) | **3 NPU calls**: gate_up_gemv ELF + silu_mul + gemv_down, then add |
| NPU dispatches for FFN | 1 (fused) | 4 (gate_up + silu + down + add) |
| SiLU x mul | NPU (inside fused op) | NPU (separate kernel, [8,1] herd) |
| Intermediates | Stay on NPU between entries | Round-trip through DDR between calls |
| IRON latency | 4766us per block | ~5.2ms per block (gate_up + silu + down + add) |

IRON keeps FFN intermediates on-device; AIR reads them back to host between each call. This adds ~0.5ms overhead per block but keeps the code simpler and the compiler happy.

### 2. Multi-Launch Merging

| Aspect | IRON | AIR |
|---|---|---|
| Q+K+V GEMVs | 3 separate dispatches | **1 ELF** with 3 launches |
| O GEMV + Add | 2 separate dispatches | **1 ELF** with 2 launches |
| Gate + Up GEMVs | Fused in SwiGLU | **1 ELF** with 2 launches |
| Technique | Runlist (xclbin-level) | Text-based MLIR stitching (ELF-level) |

AIR's multi-launch approach saves 4 dispatches per block (8→4 for attention+O+QKV), partially compensating for the decomposed FFN.

### 3. GEMV Kernel Architecture

| Aspect | IRON | AIR |
|---|---|---|
| Data path | DDR -> L1 direct (ObjectFIFO) | DDR -> L2 -> L1 (MemTile staging) |
| BD pattern | 2 BDs per FIFO (auto-increment) | Individual BDs per DMA transfer |
| Weight layout | Row-major, streamed via ObjectFIFO | Row-major, staged through L2 |
| Achieved BW | 21-51 GB/s | 22-40 GB/s |

### 4. Weight Management

| Aspect | IRON | AIR |
|---|---|---|
| Weight storage | Static BOs, pre-loaded at init | Static BOs per layer via `bo_key` + `static_input_indices` |
| Weight sync | `bo.sync()` only (no write per call) | First token: `bo.map()` write; subsequent: skipped |
| Weight transpose | Done at model init | Done at model init (`np.ascontiguousarray(W.T)`) |
| BO isolation | Per-layer BOs | Per-layer BOs via `bo_key` (e.g. `qkv_gemv_L0`, `qkv_gemv_L1`) |

Both IRON and AIR write weights once and reuse. AIR uses `bo_key` naming to isolate BOs per layer while sharing XRT hardware contexts (8 contexts, 128 BO sets).

### 5. Buffer I/O

| Aspect | IRON | AIR |
|---|---|---|
| Input/output I/O | `bo.map()` zero-copy | `bo.map()` zero-copy |
| Overhead per dispatch | ~0.1-0.5ms | ~0.8ms (Python overhead) |
| Total dispatches/block | ~12 | 10 |

Both use `bo.map()` for zero-copy buffer access. AIR's remaining overhead is Python-level (numpy reshaping, function call overhead, etc.).

### 6. Final Layers

| Aspect | IRON | AIR |
|---|---|---|
| Final RMSNorm | NPU | CPU (~0.1ms) |
| LM Head | NPU GEMV (128256x2048, 8 cols, 9.4ms) | CPU matmul (~50ms) |

---

## NPU Invocations Per Block

| IRON (~12 dispatches) | AIR (10 dispatches) |
|---|---|
| 1. RMSNorm | 1. RMSNorm |
| 2. Q GEMV | 2. **QKV GEMV** (3-launch ELF) |
| 3. K GEMV | |
| 4. V GEMV | |
| 5. RoPE Q | 3. RoPE Q |
| 6. RoPE K | 4. RoPE K |
| 7. (CPU attention) | — (CPU attention) |
| 8. O GEMV | 5. **O GEMV + Add** (2-launch ELF) |
| 9. Residual Add | |
| 10. RMSNorm | 6. RMSNorm |
| 11. **SwiGLU fused** (5 entries) | 7. **Gate+Up GEMV** (2-launch ELF) |
| 12. Residual Add | 8. SiLU x mul |
| | 9. Down GEMV |
| | 10. Residual Add |

AIR has fewer total dispatches (10 vs 12) thanks to multi-launch merging, but more FFN calls (4 vs 1) due to decomposed SwiGLU.

---

## Performance Comparison

| Metric | IRON | AIR | Notes |
|---|---|---|---|
| **NPU kernel time** | 132ms/token | **~126ms/token** | AIR slightly faster |
| **End-to-end per token** | 370ms | **351ms** | **AIR 5% faster** |
| **Dispatches per block** | ~12 | 10 | AIR fewer (multi-launch) |
| **Total dispatches/token** | ~194 | 160 | |
| **First token overhead** | ~370ms | ~893ms | AIR: weight BO init |

**AIR is faster overall** despite the decomposed FFN, because:
1. Multi-launch merging reduces dispatch count below IRON's
2. Static weight BOs eliminate per-token weight transfer
3. GEMV kernels are competitive (1.0-1.4x of IRON at kernel level)
4. The decomposed GEMVs avoid IRON's fused SwiGLU latency (4.8ms/block)

---

## Remaining Gap to Close

| Action | Expected Impact |
|---|---|
| **NPU LM Head** | ~40ms/token (replace 50ms CPU with ~10ms NPU) |
| **FFN full merge** (5 launches in 1 ELF) | ~10ms/token (3 fewer dispatches × 16 layers) |
| **Multi-tile RMSNorm** (8 cols) | ~4ms/token |
| **NPU prefill** for KV cache | Faster init (16s → 2s) |
