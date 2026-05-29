# Plan 2 (full decode) ablation report

- current_pos: **7** (after a 7-token prefill)
- trials per cell: **5** (drop trial 1 as warmup, median of remaining)
- per timed trial: ONE decode token through 16 layers + LM head + argmax

## Per-token total wall time

| Cell | Median | Range | Δ vs prev | Speedup vs prev |
|------|--------|-------|-----------|-----------------|
| **A** Naive no-merge | 256.69 ms | [256.20 ms, 257.89 ms] | — | (baseline) |
| **B** + per-layer weight BOs (#2) | 116.92 ms | [114.71 ms, 117.73 ms] | +139.77 ms | 2.20× |
| **C** + shared intermediate BOs (#3) | 113.77 ms | [112.95 ms, 114.30 ms] | +3.15 ms | 1.03× |
| **D** + multi-launch merging (#1) [production] | 90.65 ms | [90.57 ms, 90.69 ms] | +23.12 ms | 1.26× |

**A → D total speedup: 2.83×**

## Per-kernel-group medians (single call)

| Cell | rms_gemv_rope median | o_gemv_ffn median |
|------|----------------------|-------------------|
| A | 2.40 ms | 12.45 ms |
| B | 1.48 ms | 4.62 ms |
| C | 1.44 ms | 4.51 ms |
| D | 0.87 ms | 3.67 ms |

## Component breakdown (Cell D, fixed costs)

- CPU attention floor (sum across 16 layers): **3.68 ms**
- LM head (production-merged, invariant): **13.62 ms**
- Total per-token wall: **90.65 ms**

## Validation

| Cell | Validation |
|------|------------|
| A | PASS |
| B | PASS |
| C | PASS |
| D | PASS |
