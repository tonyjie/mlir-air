# LLAMA-3.2-1B BF16 on MLIR-AIR (NPU2) — Plan & Status

## Context

**Goal**: Functionally correct and performant LLAMA-3.2-1B BF16 inference on MLIR-AIR targeting NPU2 (AIE2P).

**LLAMA-3.2-1B config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_groups=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

---

## Performance Summary

| Phase | AIR | IRON | Notes |
|-------|-----|------|-------|
| **Prefill** (seq_len=2048) | **1.30s kernel / 1.54s wall** | 2.744s | **2.1x faster (kernel)** |
| **Decode** (steady-state) | **92ms/token (10.8 tok/s)** | 370ms/token | **4.0x faster** |

---

## Prefill — Per-Block Kernel Sequence (3 invocations/layer)

| # | Operation | Kernel | Launches | Avg Time |
|---|-----------|--------|----------|----------|
| 1 | RMSNorm + Q/K/V GEMMs + RoPE Q+K | `rms_gemms_rope` | 6 | 9ms |
| 2 | Flash Attention GQA | `flash_attn` | 1 | 22ms |
| 3 | O GEMM + Residual Add + FFN | `o_ffn` | 8 | 49ms |

After 16 blocks: Final RMSNorm + LM Head (8-launch ELF, 171ms).

All kernels use multi-launch ELF format. `bo.map()` zero-copy for all reads/writes.
`intermediate_indices` skips unnecessary BO syncs for kernel-overwritten buffers.

## Decode — Per-Block Kernel Sequence (2 invocations/block)

| # | Operation | Kernel | Launches | Avg Time |
|---|-----------|--------|----------|----------|
| 1 | RMSNorm + QKV GEMV + RoPE Q+K | `rms_gemv_rope` | 6 | 0.9ms |
| — | CPU attention | — | — | ~0.3ms |
| 2 | O GEMV + Add + FFN | `o_gemv_ffn` | 8 | 3.7ms |

After 16 blocks: Final RMSNorm (CPU) + NPU LM Head GEMV (8-partition, 13.5ms).

Multi-launch merges: 10 → 2 invocations/block + 1 LM Head.
External kernel rename (`-D` preprocessor defines) resolves K=2048/K=8192 GEMV type mismatch.
`intermediate_indices` skips unnecessary BO syncs for overwritten buffers.
NPU LM Head GEMV: 8-partition multi-launch (corr=0.999997 vs CPU F32).

---

## Implementation Status

- [x] Phase 0: Infrastructure & Weight Loading
- [x] Phase 1: Validate Individual Kernels at LLAMA Scale — ALL PASSED
- [x] Phase 2: Single Transformer Block — VERIFIED
- [x] Phase 3: Full 16-Layer Model — VERIFIED (Top-1 "Paris", corr 0.993)
- [x] Phase 4: Prefill Performance Optimization — COMPLETE
  - Multi-launch merges: 10 → 3 invocations/layer
  - NPU LM Head: 8-partition multi-launch (171ms vs IRON 217ms)
  - `bo.map()` zero-copy for all kernels
  - Static weight BO pre-loading for LM Head
  - 8-tile RMSNorm (broadcast DMA bug fixed): 6ms → 0.9ms standalone
  - 8-tile RoPE (row-parallel): rope_qk 11ms → 4ms per layer
  - Seq-first layout: RoPE + FlashAttention accept seq-first, zero host transposes
  - RMSNorm+QKV+RoPE 6-launch merge: 5 → 4 invocations/layer (collapse_shape inside RoPE launch)
  - O+FFN 8-launch merge: 4 → 3 invocations/layer (2D→2D collapse_shape for shared res1 buffer)
  - FlashAttention cannot be merged (channel/cascade routing blocker in aircc — see docs/plans/multi-launch/)
  - `intermediate_indices`: skip unnecessary BO syncs for kernel-overwritten buffers
  - Weight pre-loading: per-layer BOs with static_input_indices (14%→4% BO overhead)
  - **Result: 1.30s kernel / 1.54s wall vs IRON 2.744s (2.1x faster)**
- [x] Phase 5: Decode Phase — OPTIMIZED PIPELINE
  - GEMV kernel: 8-column multi-herd, 1.0-1.4x of IRON at all LLAMA shapes
  - Multi-launch merges: 10 → 2 invocations/block
  - rms_gemv_rope (6 launches): RMS+QKV+RoPE in one ELF
  - o_gemv_ffn (8 launches): O+Add+RMS+FFN in one ELF (extern kernel rename for K=8192)
  - `intermediate_indices` optimization: skip unnecessary BO syncs
  - NPU LM Head GEMV: 8-partition multi-launch (13.5ms vs CPU 258ms, 18.6× speedup)
  - Verified: corr=0.999997 vs CPU F32, top-5 token match
  - **Result: 92ms/token (10.8 tok/s) vs IRON 370ms (4.0x faster)**

- [x] Phase 6: Code Reorganization & Unified Pipeline — COMPLETE
  - Created `kernel_builder/` package (stitching, gemm_builder, cache, external_kernels)
  - Unified inference script: `llama3_inference.py` (NPU prefill + NPU decode)
  - All external C++ kernels compiled from source (no stale .o copies)
  - Moved superseded builders to `multi_launch_builder/superseded/`
  - Makefile with `make compile`, `make run`, `make profile`, `make verify`

### Remaining Optimization Opportunities

- Variable-length input sequence (currently fixed at seq_len=2048, short prompts waste compute)
- Add temperature + top-k sampling (greedy argmax causes repetition in base model)
- Prefill FFN GEMM tiling optimization (Gate/Up/Down GEMMs = 54% of prefill)
- Per-launch `omit_pingpong` control (compiler enhancement)
- NPU decode attention for long context (CPU attention grows linearly with pos)

---

## Quick Reference

```bash
cd programming_examples/llama3

make compile                    # Compile all kernels (~4 min, one-time)
make run                        # Run inference (100 tokens)
make profile                    # Run with profiling
make verify                     # Run with CPU verification
make clean                      # Remove all build artifacts
```

See `docs/usage.md` for full command reference.
See `docs/explain.md` for implementation details.
See `docs/profile.md` for performance breakdown.
