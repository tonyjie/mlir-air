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
| 7 | Flash Attention GQA | Q(2048,32,64), K(2048,8,64), V(2048,8,64) | flash_attention/kernel_fusion | **BUGGY** — CPU fallback via `attention_reference()` |
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
- [x] Phase 2: Single Transformer Block -- VERIFIED (all 15 steps corr>0.999, top-1 matches CPU)
- [x] Phase 3: Full 16-Layer Model -- RUNS END-TO-END (240 kernel invocations, no crashes)
  - BF16 output run: top-1 incorrect ("def") due to BF16 accumulator truncation
  - Fixed: F32 GEMM output eliminates truncation. Down GEMM: 0.948 -> 0.9998 per layer
  - Single-layer re-verified: all 15 steps corr>0.999. Top-1 matches CPU.
  - Kernel caching implemented: `KernelCache` compiles 10 unique kernels once, saves to `kernel_cache/`, reuses via `XRTCompileArtifact`. CLI: `--compile-only`, `--run-only`, `--profile`.
  - **16-layer profiled**: 18.7s NPU kernel time + CPU attention (334s one-time compile). Per-layer avg 1.62s.
  - Flash attention NPU kernel has correctness bug (corr=0.31 vs standard attention). GitHub issue submitted upstream.
  - See `LLAMA_flash_attention.md` for full investigation.
- [x] Phase 3A: CPU Attention Fallback -- **VERIFIED CORRECT**
  - `--cpu-attn` flag (default: on) replaces NPU flash attention with `attention_reference()` from CPU
  - **1-layer**: All 14 NPU kernels corr>0.999 (`[OK]`). Top-1 matches CPU.
  - **16-layer**: Top-1 = " Paris" (correct answer, prob=0.48). Logits corr=0.972 vs CPU F32.
  - All per-kernel, per-layer, and full-model verification tests pass.
  - Use `--npu-attn` to switch back to NPU flash attention kernel (when fixed).
- [x] Phase 3B: Fix Flash Attention NPU kernel — **FIXED** (2026-03-26)
  - All configs pass with corr > 0.996. LLAMA causal: corr=0.9976, 15ms, 2× faster than IRON.
  - See `docs/kernels/flash_attention.md`.
- [ ] Phase 4: Performance Optimization — IN PROGRESS
  - [x] Eltwise add: BF16 vec16 [8,1] → 415 µs. Matches IRON. PR #1431.
  - [x] XRT context + BO reuse: NPU kernel 18.67s → **6.49s** (65% reduction).
  - [x] GEMM: Optimized tiles + 8×4 herd + BF16 output **integrated**. NPU 6.49s → **3.60s**.
  - [x] SwiGLU: [8,1] herd + 16-wide vectors. 59ms → 37ms.
  - [x] FlashAttention: **Fixed and 2× faster than IRON** (15ms vs 31ms).
  - [x] FlashAttention integration: **DONE** (2026-03-26). NPU attention is now default. Top-1 " Paris", logits corr=0.993. Wall time ~44s → ~4-5s. NPU kernel total 3.60s.
  - **Next**: RoPE/RMSNorm vectorization, FFN fusion.
  - See `docs/performance_optimization.md` for full breakdown and roadmap.
- [ ] Phase 5: Decode Phase (future work)

## Files

```
programming_examples/llama3/
  llama3_prefill.py          # NPU integration (KernelCache + transformer block pipeline)
  llama3_weights.py          # Weight loading from safetensors + RoPE LUT
  llama3_reference.py        # CPU reference implementation (F32)
  swiglu_activation.py       # Standalone SwiGLU AIR kernel (Python)
  swiglu_activation.cc       # SwiGLU C++ kernel (for Peano)
  Makefile                   # Build targets
  docs/
    LLAMA_PLAN.md                   # This plan
    LLAMA_progress.md               # Session log
    LLAMA_verification.md           # Architecture, commands, test results, bugs
    LLAMA_explanation.md            # Code walkthrough
    performance_optimization.md     # Performance profiling & optimization roadmap
    LLAMA_flash_attention.md        # Flash attention bug investigation
    LLAMA_gemm.md                   # GEMM precision investigation (historical)
    kernels/
      eltwise_add.md                # Eltwise add optimization (herd sweep, IRON comparison)
      gemm.md                       # GEMM precision & performance analysis
```
