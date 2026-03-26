# Flash Attention Kernel — Status & Analysis

## Role in LLAMA Pipeline

Step 7 of each transformer block: Multi-head attention with GQA and causal masking.

| Parameter | LLAMA-3.2-1B Value |
|-----------|-------------------|
| Sequence length (LQ=LK) | 2048 |
| Head dimension (DK=DV) | 64 |
| Number of Q heads | 32 |
| Number of KV heads | 8 (GQA ratio 4:1) |
| Causal masking | Yes |
| Tile sizes | LQP=256, LKP=64 |

---

## Current Status (2026-03-26): Integrated into LLAMA Pipeline

### FlashAttention is Working — Integrated and Verified

**Standalone test** (`make run`): corr=0.9976 for LLAMA causal config (32Q/8KV, seq_len=2048).

**LLAMA pipeline integration** (16 layers, all on NPU):

| Metric | Value |
|--------|-------|
| Top-1 prediction | " Paris" (correct) |
| Logits correlation vs CPU F32 | **0.993** |
| Per-step correlation | >0.999 (all 240 invocations) |
| NPU kernel time (flash_attn) | 0.46s total (29ms avg per layer) |
| NPU kernel time (all kernels) | 3.60s total |

### Integration Notes

- Kernel expects **unscaled Q** — scaling by 1/sqrt(dk) is handled internally by the kernel
- Kernel takes **4 args** (Q, K, V, Output) — no mask buffer (causal masking is internal)
- `test_precision.py` is out of date (passes 5 args for old kernel interface) — use `make run` instead

### Performance

| Metric | AIR FlashAttention | IRON MHA | AIR vs IRON |
|--------|-------------------|----------|-------------|
| Latency (standalone) | **15,022 µs** (15ms) | 30,989 µs (31ms) | **2.06× faster** |
| Latency (in pipeline) | **29,000 µs** (29ms) | — | — |
| GFLOP/s | **2,281** | ~1,100 | **2.1× faster** |
| corr (standalone) | 0.9976 | 0.9976 | **Matched** |

### Tile Config Analysis (LLAMA causal, 32h/8kv, 2048 seq)

Only **LQP=256, LKP=64** works for causal mode. Swept all valid combinations:

| LQP | LKP | Status | Constraint |
|-----|-----|--------|-----------|
| * | 32 | FAIL | Causal doesn't support DV tiling (LKP < DK) |
| 64 | 64 | FAIL | Causal requires `tile_size_q == LKP` (16 ≠ 64) |
| 128 | 64 | FAIL | Same (32 ≠ 64) |
| **256** | **64** | **PASS** | **tile_size_q = 256/4 = 64 == LKP ✓** |
| 512 | 64 | FAIL | Same constraint (128 ≠ 64) |
| 1024 | 64 | FAIL | Same (256 ≠ 64) |
| * | 128+ | FAIL | DK (64) must be divisible by LKP |

Three constraints intersect at exactly one point:
1. **LKP ≤ DK** → LKP ∈ {32, 64}
2. **LKP ≥ DK** (no DV tiling for causal) → LKP ≥ 64
3. **tile_size_q == LKP** (causal requirement) → LQP/4 = 64 → LQP = 256

---

## Reproducible Commands

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Correctness (LLAMA causal config)
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
    NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# Performance profiling
make profile LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 \
    NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# Precision analysis (correlation-based)
python3 test_precision.py --num-heads 32 --num-kv-heads 8 --causal
```

---

## Historical Investigation (resolved)

### Previous Bug: corr=0.13-0.34 (before 2026-03-26 fix)

The kernel produced output with **no correlation** to standard attention for all configs at LQ=2048. `make run PASS` was a false positive — element-wise tolerance (`atol=0.15, rtol=0.04`) couldn't detect wrong attention patterns because output values were in the correct range (~1.5) but mathematically incorrect.

Fixed in upstream update (2026-03-26). Correlation now 0.996-0.998 for all configs.

### Earlier Bug: corr=0.31 (before PR #1438)

Original flash attention kernel with missing causal masking and BD exhaustion. Fixed by PR #1438 (causal BD exhaustion) and subsequent upstream fixes.

---

## Related Documents

- `performance_optimization.md` — Overall LLAMA optimization roadmap
- `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md` — IRON MHA profiling
- `kernels/gemm.md` — GEMM optimization (similar precision analysis approach)
