# LLAMA-3.2-1B Decode on MLIR-AIR (NPU2) -- Plan & Status

## Context

Prefill (seq_len=2048) is complete on AIR: 1.84s (33% faster than IRON).
Decode pipeline is operational: **351ms/token (5% faster than IRON's 370ms)**.

**IRON baseline**: 2.94 tok/s, 370 ms/token.
**AIR current**: 2.81 tok/s, 351 ms/token (steady state).

---

## Per-Block Kernel Sequence (Current — 10 NPU invocations/block)

| # | Operation | Kernel | Herd | Status |
|---|-----------|--------|------|--------|
| 1 | RMSNorm (pre-attn) | `rmsnorm` xclbin | [1,1] | DONE |
| 2 | Q+K+V GEMV | `qkv_gemv` ELF (3 launches) | [8,1]×3 | DONE |
| 3 | RoPE Q | `rope_q` xclbin | [1,1] | DONE |
| 4 | RoPE K | `rope_k` xclbin | [1,1] | DONE |
| — | Attention | CPU (numpy) | — | DONE |
| 5 | O GEMV + Add | `o_gemv_add` ELF (2 launches) | [8,1]+[8,1] | DONE |
| 6 | RMSNorm (pre-FFN) | `rmsnorm` xclbin | [1,1] | DONE |
| 7 | Gate+Up GEMV | `gate_up_gemv` ELF (2 launches) | [8,1]×2 | DONE |
| 8 | SiLU x mul | `silu_mul` xclbin | [8,1] | DONE |
| 9 | Down GEMV | `gemv_down` xclbin | [8,1] | DONE |
| 10 | Residual Add | `add` xclbin | [8,1] | DONE |

After 16 blocks: Final RMSNorm (CPU) + LM Head matmul (CPU).

---

## Implementation Phases — ALL COMPLETE

### Phase 1: GEMV Kernel — DONE
- [x] Multi-column GEMV (herd_m=8) at all LLAMA shapes
- [x] Broadcast DMA bug fixed (mlir-air rebuild)
- [x] Optimal tile configs: tile_m=8/m_input=4 (K=2048), tile_m=2/m_input=1 (K=8192)
- [x] Backend flags: `runtime_loop_tiling_sizes=[16,16]`, `omit_pingpong=''`, `lock_fix=False`
- [x] Profiled vs IRON: 0.8-1.4x at all shapes (see `docs/kernels/gemv.md`)

### Phase 2: Decode-Size Elementwise Ops — DONE
- [x] RMSNorm at M=1 N=2048, [1,1] herd
- [x] Eltwise Add at n=2048, [8,1] herd
- [x] RoPE at (32,64) and (8,64), [1,1] herd
- [x] SiLU×mul at n=8192, [8,1] herd (NPU)

### Phase 3: Multi-Launch Merges — DONE (partial)
- [x] Q+K+V GEMV merged (3 launches in 1 ELF)
- [x] O GEMV + Add merged (2 launches in 1 ELF)
- [x] Gate + Up GEMV merged (2 launches in 1 ELF)
- [ ] FFN full merge (Gate+Up+SiLU+Down+Add) — BLOCKED by linalg_fill type mismatch

### Phase 4: Decode Pipeline — DONE
- [x] KV cache management (CPU-managed numpy arrays)
- [x] CPU attention (GQA with KV cache lookup)
- [x] CPU prefill for KV cache population
- [x] End-to-end decode loop with per-token timing

### Phase 5: Performance Optimization — DONE
- [x] Static weight BOs (`bo_key` per layer, `static_input_indices`)
- [x] `bo.map()` zero-copy for all BO reads/writes
- [x] Pre-transposed weights (one-time init)
- [x] GEMV kernel tuning (tile sizes, backend flags)
- [x] Result: **351ms/token (5% faster than IRON)**

---

## Remaining Opportunities

| Priority | Action | Expected Impact | Blocker |
|---|---|---|---|
| 1 | NPU LM Head GEMV (128256×2048) | ~40ms/token | None — straightforward |
| 2 | FFN full merge (5 launches) | ~10ms/token | linalg_fill memref type mismatch |
| 3 | Multi-tile RMSNorm ([8,1]) | ~4ms/token | aiecc weight broadcast DMA bug |
| 4 | NPU prefill for KV cache | Faster init (16s→2s) | Need to integrate NPU prefill pipeline |
| 5 | Unified inference script | User convenience | None |

---

## IRON Reference

Detailed IRON profiling data: `decode/iron_decode_reference.md`
IRON source code: `/home/jiajli/apps/IRON/iron/operators/gemv/`
IRON GEMV kernel: `/home/jiajli/apps/IRON/aie_kernels/generic/mv.cc`
