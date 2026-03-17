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

Tolerance: `atol=0.15, rtol=0.04`. All at LQ=LK=2048, LQP=256, LKP=64, DK=DV=64, val_range=3.

| Config | NH | NKV | Causal | max_diff | avg_diff | Mismatches | Total | Fail% | Status |
|--------|----|-----|--------|---------|---------|-----------|-------|-------|--------|
| LIT causal | 2 | 2 | Yes | — | — | 0 | 32K | 0% | **PASS** (LQ=16384) |
| LIT GQA | 12 | 6 | No | — | — | 0 | 393K | 0% | **PASS** (LQ=512) |
| **2 heads** | **2** | **2** | **No** | **0.469** | **0.306** | **5,817** | **262K** | **2.2%** | **FAIL** |
| 4 heads | 4 | 4 | No | — | — | 0 | 524K | 0% | **PASS** |
| 6 heads | 6 | 6 | No | — | — | 0 | 786K | 0% | **PASS** |
| 8 heads MHA | 8 | 8 | No | — | — | 0 | 1.0M | 0% | **PASS** |
| 8 heads GQA | 8 | 4 | No | — | — | 0 | 1.0M | 0% | **PASS** |
| **10 heads** | **10** | **10** | **No** | — | — | **39,704** | **1.3M** | **3.0%** | **FAIL** |
| **12 heads** | **12** | **12** | **No** | **0.305** | **0.239** | **43,924** | **1.6M** | **2.8%** | **FAIL** |
| **LLAMA** | **32** | **8** | **No** | **0.289** | **0.241** | **85,792** | **4.2M** | **2.0%** | **FAIL** |
| **LLAMA causal** | **32** | **8** | **Yes** | **0.477** | **0.286** | **357,646** | **4.2M** | **8.5%** | **FAIL** |
| **IRON MHA** | **32** | **8** | **Yes** | **0.25** | **0.021** | **~110** | **~21K** | **0.1%** | **PASS** |

### Failure Pattern

The pattern is **non-monotonic**: NH=2 fails, NH=4/6/8 pass, NH≥10 fails. This suggests the issue is related to how the kernel partitions heads into the AIE herd/segment, not simply "more heads = worse."

- **NH=4,6,8**: These divide evenly into the kernel's internal segment unroll factor (2) and herd configuration, producing correct results.
- **NH=2**: Likely a boundary case in the segment partitioning.
- **NH≥10**: Exceeds some internal buffer or DMA scheduling limit.

The LIT tests pass because they use different LQ/LK values (16384 or 512) where the total buffer sizes are within limits despite having many heads.

The failures are **precision mismatches** (max_diff=0.24-0.48), not compilation errors. Causal masking adds more errors (8.5% vs 2.0%) due to additional masking computation. IRON MHA achieves much better precision (0.1% fail, max_err=0.25) at the same LLAMA scale.

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
