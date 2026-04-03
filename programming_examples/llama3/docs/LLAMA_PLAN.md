# LLAMA-3.2-1B BF16 on MLIR-AIR (NPU2) — Plan & Status

## Context

**Goal**: Functionally correct and performant LLAMA-3.2-1B BF16 inference on MLIR-AIR targeting NPU2 (AIE2P).

**LLAMA-3.2-1B config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_groups=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

---

## Performance Summary

| Phase | AIR | IRON | Notes |
|-------|-----|------|-------|
| **Prefill** (seq_len=2048) | **1.92s** | 2.744s | **30% faster** |
| **Decode** (steady-state) | **351ms/token** | 370ms/token | **5% faster** |

---

## Prefill — Per-Block Kernel Sequence (5 invocations/layer)

| # | Operation | Kernel | Launches | Avg Time |
|---|-----------|--------|----------|----------|
| 1 | RMSNorm + Q/K/V GEMMs | `rms_attn_gemms` | 4 | 9ms |
| 2 | RoPE Q+K | `rope_qk` | 2 herds | 11ms |
| 3 | Flash Attention GQA | `flash_attn` | 1 | 20ms |
| 4 | O GEMM + Residual Add | `o_proj_add` | 2 | 6ms |
| 5 | RMSNorm + FFN + Residual Add | `ffn_full` | 6 | 52ms |

After 16 blocks: Final RMSNorm + LM Head (8-launch ELF, 171ms).

All kernels use multi-launch ELF format. `bo.map()` zero-copy for all reads/writes. RMSNorm uses 8-tile herd (broadcast weight DMA).

## Decode — Per-Block Kernel Sequence (10 invocations/block)

| # | Operation | Kernel | Herd | Avg Time |
|---|-----------|--------|------|----------|
| 1 | RMSNorm (pre-attn) | `rmsnorm` xclbin | [1,1] | 0.3ms |
| 2 | Q+K+V GEMV | `qkv_gemv` ELF (3 launches) | [8,1]×3 | 1.0ms |
| 3 | RoPE Q | `rope_q` xclbin | [1,1] | 0.3ms |
| 4 | RoPE K | `rope_k` xclbin | [1,1] | 0.2ms |
| — | CPU attention | — | — | ~2ms |
| 5 | O GEMV + Add | `o_gemv_add` ELF (2 launches) | [8,1]+[8,1] | 0.6ms |
| 6 | RMSNorm (pre-FFN) | `rmsnorm` xclbin | [1,1] | 0.3ms |
| 7 | Gate+Up GEMV | `gate_up_gemv` ELF (2 launches) | [8,1]×2 | 2.5ms |
| 8 | SiLU×mul | `silu_mul` xclbin | [8,1] | 0.3ms |
| 9 | Down GEMV | `gemv_down` xclbin | [8,1] | 2.1ms |
| 10 | Residual Add | `add` xclbin | [8,1] | 0.3ms |

After 16 blocks: Final RMSNorm + LM Head (CPU, ~50ms).

---

## Implementation Status

- [x] Phase 0: Infrastructure & Weight Loading
- [x] Phase 1: Validate Individual Kernels at LLAMA Scale — ALL PASSED
- [x] Phase 2: Single Transformer Block — VERIFIED
- [x] Phase 3: Full 16-Layer Model — VERIFIED (Top-1 "Paris", corr 0.993)
- [x] Phase 4: Prefill Performance Optimization — COMPLETE
  - Multi-launch merges: 10 → 5 invocations/layer
  - NPU LM Head: 8-partition multi-launch (171ms vs IRON 217ms)
  - `bo.map()` zero-copy for all kernels
  - Static weight BO pre-loading for LM Head
  - 8-tile RMSNorm (broadcast DMA bug fixed): 6ms → 0.9ms standalone
  - **Result: 1.92s total prefill vs IRON 2.744s (30% faster)**
- [x] Phase 5: Decode Phase — OPTIMIZED PIPELINE
  - GEMV kernel: 8-column multi-herd, 1.0-1.4x of IRON at all LLAMA shapes
  - Multi-launch merges: Q+K+V (3→1), O+Add (2→1), Gate+Up (2→1)
  - NPU SiLU×mul, 8-tile eltwise_add
  - Static weight BO caching, `bo.map()` zero-copy
  - **Result: 351ms/token vs IRON 370ms (5% faster)**

### Remaining Optimization Opportunities

- FFN full merge for decode (5 launches in 1 ELF) — blocked by memref type mismatch between GEMVs with different tile_m
- NPU LM Head for decode — currently CPU matmul (~50ms/token)
- Unified prefill + decode script

---

## Quick Reference

```bash
cd programming_examples/llama3/build_peano

# Prefill
python3 ../llama3_prefill.py --compile-only --profile       # compile (~3 min)
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --profile

# Decode
python3 ../llama3_decode.py --compile-only                   # compile (~10s)
python3 ../llama3_decode.py --run-only --n-tokens 100 --profile
```

## Files

```
programming_examples/llama3/
  llama3_prefill.py                   # Main prefill pipeline
  llama3_decode.py                    # Main decode pipeline
  llama3_weights.py                   # Weight loading + RoPE LUT
  llama3_reference.py                 # CPU F32 reference
  multi_launch_builder/               # Multi-launch ELF builders
    rms_attn_gemms_multi.py             (4 launches: RMS+Q+K+V)
    rope_qk_multi.py                    (2 herds: RoPE Q+K)
    o_proj_add_multi.py                 (2 launches: O GEMM+Add)
    ffn_full_multi.py                   (6 launches: RMS+Gate+Up+SiLU+Down+Add)
    lm_head_multi.py                    (8 launches: LM Head partitions)
    rms_qkv_gemv_multi.py              (3 launches: Q+K+V GEMV for decode)
    o_gemv_add_multi.py                 (2 launches: O GEMV+Add for decode)
    ffn_gemv_multi.py                   (2 launches: Gate+Up GEMV for decode)
  ffn_swiglu/                         # SiLU×mul kernel
  docs/
    LLAMA_PLAN.md                       # This file
    LLAMA_progress.md                   # Session log & kernel breakdown
    LLAMA_explanation.md                # Code architecture walkthrough
    perf_opt_prefill.md                 # Prefill performance history
    decode/                             # Decode documentation
    kernels/                            # Per-kernel analysis
    issues/                             # Compiler bugs & reproducers
```
