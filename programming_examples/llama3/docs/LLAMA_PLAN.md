# LLAMA-3.2-1B BF16 on MLIR-AIR (NPU2) -- Plan & Status

## Context

**Goal**: Functionally correct and performant LLAMA-3.2-1B BF16 inference on MLIR-AIR targeting NPU2 (AIE2P).

**LLAMA-3.2-1B config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_groups=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

**Current status**: Prefill complete. **25% faster than IRON** (2.05s vs 2.744s). Decode is future work.

---

## Per-Block Kernel Sequence (Current — 5 invocations/layer)

| # | Operation | Kernel | Launches | Time |
|---|-----------|--------|----------|------|
| 1 | RMSNorm + Q/K/V GEMMs | `rms_attn_gemms` | 4 | 14ms |
| 2 | RoPE Q+K | `rope_qk` | 2 herds | 11ms |
| 3 | Flash Attention GQA | `flash_attn` | 1 | 20ms |
| 4 | O GEMM + Residual Add | `o_proj_add` | 2 | 6ms |
| 5 | RMSNorm + FFN + Residual Add | `ffn_full` | 6 | 56ms |

After 16 blocks: Final RMSNorm + LM Head (8-launch ELF, 173ms).

All kernels use multi-launch ELF format. `bo.map()` zero-copy for all reads/writes.

---

## Implementation Status

- [x] Phase 0: Infrastructure & Weight Loading
- [x] Phase 1: Validate Individual Kernels at LLAMA Scale — ALL PASSED
- [x] Phase 2: Single Transformer Block — VERIFIED
- [x] Phase 3: Full 16-Layer Model — VERIFIED
  - Top-1 " Paris" for "The capital of France is"
  - Logits correlation 0.989 vs CPU F32 reference
  - NPU FlashAttention integrated (2× faster than IRON)
- [x] Phase 4: Performance Optimization — COMPLETE
  - Multi-launch merges: 10 → 5 invocations/layer
  - NPU LM Head: 8-partition multi-launch (173ms vs CPU 1526ms)
  - `bo.map()` zero-copy for all kernels
  - Static weight BO pre-loading for LM Head
  - **Result: 2.05s total prefill vs IRON 2.744s (25% faster)**
  - See `perf_opt_prefill.md` for full breakdown
- [x] Phase 5: Decode Phase — OPTIMIZED PIPELINE
  - GEMV kernel: 8-column multi-herd, 1.0-1.4x of IRON at all LLAMA shapes
  - Decode pipeline: 10 NPU invocations/block + CPU attention
  - Multi-launch merges: Q+K+V (3→1), O+Add (2→1), Gate+Up (2→1)
  - NPU SiLU×mul (moved from CPU), 8-tile eltwise_add
  - Static weight BO caching (write once, skip on subsequent tokens)
  - KV cache: CPU-managed, populated from CPU prefill
  - Correct text generation: "The capital of France is" → "Paris"
  - **~351ms/token (vs IRON 370ms) — AIR 5% faster**
  - NPU kernel time ~126ms/token (vs IRON 132ms)
  - See `decode/DECODE_EXPLANATION.md` for details

### Remaining Optimization Opportunities
- Multi-tile RMSNorm (8ms → ~4ms) — blocked by aiecc weight broadcast DMA bug
- FFN full merge (5 launches in 1 ELF) — blocked by memref type mismatch between GEMVs with different tile_m
- NPU LM Head — currently CPU matmul (~50ms/token)
- See `issues/` for compiler bug details and reproducers

---

## Files

```
programming_examples/llama3/
  llama3_prefill.py                   # Main pipeline (KernelCache + transformer block)
  llama3_weights.py                   # Weight loading from safetensors + RoPE LUT
  llama3_reference.py                 # CPU F32 reference implementation
  multi_launch_builder/               # Multi-launch ELF builders
    ffn_full_multi.py                   (6 launches: RMS+Gate+Up+SiLU+Down+Add)
    rms_attn_gemms_multi.py             (4 launches: RMS+Q+K+V)
    rope_qk_multi.py                    (2 herds: RoPE Q+K)
    o_proj_add_multi.py                 (2 launches: O GEMM+Add)
    lm_head_multi.py                    (8 launches: LM Head partitions)
    attn_gemms_multi.py                 (3 launches: Q+K+V, standalone)
  ffn_swiglu/                         # SiLU×mul kernel
    silu_and_mul.py                     (kernel builder)
    silu_and_mul.cc                     (C++ kernel)
    run.py                              (standalone FFN test)
  docs/
    LLAMA_PLAN.md                       # This file
    LLAMA_progress.md                   # Current status & session log
    LLAMA_explanation.md                # Code architecture walkthrough
    perf_opt_prefill.md         # Performance results & history
    host_optimization.md                # BO/host overhead analysis
    kernels/                            # Per-kernel analysis (6 files)
    plans/multi-launch/                 # Multi-launch optimization plans
    issues/                             # Compiler bugs & reproducers
    decode/                             # Decode phase planning
```
