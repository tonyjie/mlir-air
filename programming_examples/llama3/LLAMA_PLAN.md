# LLAMA-3.2-1B BF16 on MLIR-AIR (NPU2) -- High-Level Plan

## Context

**Goal**: Implement a functionally correct, then performant, LLAMA-3.2-1B BF16 inference on MLIR-AIR targeting NPU2 (AIE2P). Use IRON's implementation as the reference architecture.

**LLAMA-3.2-1B config**: 16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_groups=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

**Key architectural decisions**:
1. **Prefill first** (seq_len=2048). Decode (seq_len=1, GEMV-based) is a later phase.
2. **ELF output format** -- no xclbin 5-argument limit, supports multi-argument kernels.
3. **Embedding and LM Head on CPU initially** -- embedding is a lookup table; LM Head GEMM (2048x128256) is very large.
4. **Each operator as separate kernel invocation** initially -- compile each operator to its own ELF, invoke sequentially from Python. Fuse later for performance.
5. **Decompose FFN SwiGLU into separate ops** -- the existing fused `ffn_swiglu/prefill` can't scale to LLAMA dims (L1 overflow at DIM=2048). Use the tiled GEMM kernel + standalone SwiGLU activation instead.

---

## Per-Block Kernel Sequence (Prefill, seq_len=2048)

Each transformer block runs 15 kernel invocations:

| # | Operation | Shape | MLIR-AIR Kernel | Status |
|---|-----------|-------|-----------------|--------|
| 1 | RMSNorm (pre-attn) | (2048, 2048) + weight(2048,) | weighted_rms_norm | VALIDATED |
| 2 | Q Projection | (2048, 2048) @ W(2048, 2048) | matrix_multiplication/bf16 | VALIDATED |
| 3 | K Projection | (2048, 2048) @ W(2048, 512) | matrix_multiplication/bf16 | VALIDATED |
| 4 | V Projection | (2048, 2048) @ W(2048, 512) | matrix_multiplication/bf16 | VALIDATED |
| 5 | RoPE on Q | (2048*32, 64) with LUT | rope_lut | VALIDATED (seq_len=65536) |
| 6 | RoPE on K | (2048*8, 64) with LUT | rope_lut | VALIDATED |
| 7 | Flash Attention GQA | Q(2048,32,64), K(2048,8,64), V(2048,8,64) | flash_attention/kernel_fusion | VALIDATED (LQ=2048,LK=2048,LKP=64,LQP=256) |
| 8 | O Projection | (2048, 2048) @ W(2048, 2048) | matrix_multiplication/bf16 | VALIDATED |
| 9 | Residual Add | (2048*2048) + (2048*2048) | eltwise_add | VALIDATED (n=4194304) |
| 10 | RMSNorm (pre-FFN) | (2048, 2048) + weight(2048,) | weighted_rms_norm | VALIDATED |
| 11 | Gate GEMM | (2048, 2048) @ W(2048, 8192) | matrix_multiplication/bf16 | VALIDATED |
| 12 | Up GEMM | (2048, 2048) @ W(2048, 8192) | matrix_multiplication/bf16 | VALIDATED |
| 13 | SwiGLU activation | SiLU(gate) * up, n=2048*8192 | swiglu_activation | VALIDATED (n=16777216) |
| 14 | Down GEMM | (2048, 8192) @ W(8192, 2048) | matrix_multiplication/bf16 | VALIDATED |
| 15 | Residual Add | (2048*2048) + (2048*2048) | eltwise_add | VALIDATED |

After 16 blocks: Final RMSNorm + LM Head (CPU).

### Flash Attention Known Good Config

```bash
make profile LQ=2048 LK=2048 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8
```

Note: seq_len=128 had issues with Flash Attention (required LK=256 workaround). seq_len=2048 is the intended operating point.

---

## Implementation Status

- [x] Phase 0: Infrastructure & Weight Loading
- [x] Phase 1: Validate Individual Kernels at LLAMA Scale (seq_len=2048) -- ALL PASSED
  - [x] 1A. GEMM (4 shapes at M=2048: Q/O, K/V, Gate/Up, Down)
  - [x] 1B. Weighted RMSNorm at M=2048, N=2048
  - [x] 1C. RoPE LUT at seq_len=65536 (2048*32), embed_dim=64
  - [x] 1D. Eltwise Add at n=4194304
  - [x] 1E. Flash Attention GQA (LQ=2048, LK=2048, LKP=64, LQP=256, 32Q/8KV)
  - [x] 1F. SwiGLU Activation at n=16777216
- [x] Phase 2: Single Transformer Block -- VERIFIED (all 15 steps corr>0.948, top-1 matches CPU)
- [ ] Phase 3: Full 16-Layer Model (run with real weights)
- [ ] Phase 4: Performance Optimization
- [ ] Phase 5: Decode Phase (future work)

### Previous validation (seq_len=128) -- kept for reference

All kernels passed at seq_len=128 dimensions (2026-03-12). These are smaller shapes that served as initial smoke tests but are not the target operating point.

## Files Created

```
programming_examples/llama3/
  llama3_weights.py          # Weight loading from safetensors + RoPE LUT
  llama3_reference.py        # CPU reference implementation (F32)
  llama3_prefill.py          # NPU integration: sequential kernel invocations
  swiglu_activation.py       # Standalone SwiGLU AIR kernel (Python)
  swiglu_activation.cc       # SwiGLU C++ kernel (for Peano)
  Makefile                   # Build targets
  run_npu2_swiglu_peano.lit  # LIT test for SwiGLU
  LLAMA_PLAN.md              # This plan
  LLAMA_progress.md          # Progress tracker (session log)
  LLAMA_verification.md      # Commands, test results, bugs
  LLAMA_explanation.md       # Code walkthrough (architecture -> implementation)
```
