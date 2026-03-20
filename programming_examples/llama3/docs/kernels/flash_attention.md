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

**Current LLAMA workaround**: CPU fallback via `attention_reference()` (`--cpu-attn`, default).

---

## Current Status (2026-03-20, after upstream fix PR #1438)

### What's Fixed

PR #1438 fixed the **causal masking BD exhaustion** bug and the **kernel correctness** for supported configurations. The original issue (corr=0.31 vs standard attention) is resolved for configs within the supported range.

### Supported vs Unsupported Configurations

Tested with `atol=0.15, rtol=0.04` (the kernel's built-in tolerance).

### Precision Metrics (measured with `test_precision.py`)

All at LQ=LK=2048, LQP=256, LKP=64, DK=DV=64, val_range=3.

| Config | NH | NKV | Causal | corr | max_err | mean_err | 4%-fail | `make run` |
|--------|----|-----|--------|------|---------|----------|---------|-----------|
| 4h MHA | 4 | 4 | No | **0.132** | 0.719 | 0.073 | 10.1% | PASS* |
| 8h MHA | 8 | 8 | No | **0.175** | 1.217 | 0.076 | 11.3% | PASS* |
| **LLAMA** | **32** | **8** | **No** | **0.287** | **0.711** | **0.064** | **6.5%** | **FAIL** |
| **LLAMA causal** | **32** | **8** | **Yes** | **0.343** | **2.160** | **0.096** | **17.9%** | **FAIL** |
| **IRON MHA** | **32** | **8** | **Yes** | **0.998** | **0.25** | **0.021** | **0.1%** | **PASS** |

\*`make run` reports PASS because `max_mismatch_percentage=2` allows up to 2% element failures, and mean_err (~0.07) is below `atol=0.15`. However the **correlation is 0.13-0.18** — the output is numerically plausible but mathematically wrong.

### Critical Finding

**The flash attention kernel correctness bug is NOT fixed.** Correlation = 0.13-0.34 for ALL configs at LQ=2048, including those that `make run` reports as PASS. This is the same magnitude as the original corr=0.31 bug.

The `make run PASS` is misleading because:
- Output values are in the correct **range** (~1.0-2.0, softmax-averaged V values)
- **Mean absolute error** is small (~0.07) relative to `atol=0.15`
- But the output has **no correlation** with the correct attention — it's computing the wrong thing
- The test's element-wise tolerance (`atol=0.15, rtol=0.04, max_mismatch_percentage=2`) cannot detect this class of bug where the output is in-range but mathematically incorrect

**IRON MHA achieves corr=0.998 at the same scale**, confirming the computation is achievable.

### Why `make run` Passes for Some Configs

With `max_mismatch_percentage=2` and `atol=0.15`:
- Output mean ≈ 1.5 (correct range for averaged V in [0,3])
- mean_err ≈ 0.07 < atol=0.15 → most elements pass
- Only ~6-11% elements exceed tolerance → 4h/8h have <2% mismatches at the default `atol/rtol`
- But correlation is 0.13-0.18 → output is essentially random within the correct range

### Reproducible Precision Measurement

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Precision test (measures corr, max_err, mean_err, 4%-fail)
python3 test_precision.py --num-heads 8 --num-kv-heads 8
python3 test_precision.py --num-heads 32 --num-kv-heads 8 --causal
```

### Likely Root Cause

The precision failure at high head counts is likely related to:
1. **BFP16 rounding accumulation** — same issue as GEMM (floor rounding in `aievec.srs`). The flash attention kernel performs Q@K and Attn@V matmuls using BFP16 emulation internally.
2. **Buffer size / BD allocation at scale** — 32 heads × 2048 seq × 64 dim generates large intermediate buffers that may cause DMA scheduling issues.
3. **Cascade merge precision** — the 4-stage cascade merge for flash attention introduces additional rescaling steps that accumulate BFP16 rounding errors.

---

## IRON Reference

IRON MHA at LLAMA scale (from `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md`):

| Metric | IRON MHA |
|--------|---------|
| Shape | 2048 seq, 32 heads, 64 dim, 8 pipelines |
| Latency (standalone) | 30,989 µs |
| corr | 0.99758 |
| max_err | 0.25 |
| 4%-fail | 0.1% |
| Model avg | 37.0 ms |

IRON achieves corr=0.99758 at the full LLAMA scale. This confirms the computation is achievable on NPU2 — the AIR flash attention kernel needs fixes for large head counts.

---

## Historical Investigation (before PR #1438 fix)

### Original Bug: corr=0.31 vs Standard Attention

The pre-fix flash attention kernel produced output with only **corr=0.31** against `attention_reference()`. This was caused by:
1. Missing causal masking (`causal=False` was hardcoded)
2. BD exhaustion when enabling `causal=True`
3. Kernel correctness issues in the causal path

### Per-Kernel Diagnostic

Built `diagnose_layer.py` to isolate the problem:

```
 7  FlashAttn GQA    0.34026316    0.6926     FAIL  <<<
```

All other 14 kernels had corr>0.999 — flash attention was the sole bad kernel.

### Standalone Test False Positive

The standalone `make run PASS!` was misleading because:
- Test data in narrow range [0.5, 1.0]
- Tolerances `atol=0.5, rtol=0.2` → any output in [0, 1.5] passes
- Later fixed to `atol=0.15, rtol=0.04`

### CPU Fallback Workaround

Implemented `--cpu-attn` flag (default: on) to use `attention_reference()` from CPU. This enabled 16-layer LLAMA verification (top-1 = " Paris", correct) while waiting for the kernel fix.

---

## Configuration Sweep (historical, before PR #1438)

10 configurations tested at LLAMA shapes. Only 3 compiled and passed:

| # | LQP | LKP | Causal | Status | Failure Mode |
|---|-----|-----|--------|--------|-------------|
| 1 | 256 | 64 | No | **PASS** (2427 GFLOP/s) | — |
| 4 | 128 | 128 | No | **PASS** (573 GFLOP/s) | — |
| 6 | 256 | 32 | No | **PASS** (687 GFLOP/s) | — |
| 8 | 256 | 64 | Yes | **NOW PASS** (after PR #1438) | Was BD exhaustion |
| 2,7,9 | 128/128 | 64/32/32 | Various | FAIL | Compilation timeout |
| 3,5,10 | 512/256/512 | 64/128/128 | Various | FAIL | Pass pipeline failure |

---

## Reproducible Commands

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# LIT test configs (should PASS)
make run LK=16384 LKP=64 LQ=16384 LQP=256 DK=64 DV=64 NUM_HEADS=2 VAL_RANGE=3 EXTRA_PY_FLAGS="--causal"
make run NUM_HEADS=12 NUM_KV_HEADS=6

# Working configs at 2048 seq
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 NUM_HEADS=8 NUM_KV_HEADS=8
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 NUM_HEADS=8 NUM_KV_HEADS=4

# LLAMA config (currently FAILS — precision mismatches at >8 heads)
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 NUM_HEADS=32 NUM_KV_HEADS=8 EXTRA_PY_FLAGS="--causal"

# LLAMA pipeline with CPU attention fallback
cd programming_examples/llama3/build_peano
python3 ../llama3_prefill.py --run-only --n-layers 16 --verify --cpu-attn
```

---

## Next Steps

1. **Investigate precision at >8 heads**: Determine if the failure is from BFP16 rounding (same as GEMM issue, potentially fixable with rounding mode), buffer overflow, or cascade merge precision.
2. **Test with rounding mode fix**: The MLIR-AIE rounding mode fix may help the flash attention's internal matmuls — needs testing.
3. **Consider partial NPU offload**: Use NPU attention for ≤8 heads per dispatch, splitting 32 heads into 4 batches of 8.

---

## Related Documents

- `performance_optimization.md` — Overall LLAMA optimization roadmap
- `/home/jiajli/apps/IRON/docs/IRON_LLAMA_profile.md` — IRON MHA profiling
- `kernels/gemm.md` — GEMM rounding mode analysis (same BFP16 issue may apply)
