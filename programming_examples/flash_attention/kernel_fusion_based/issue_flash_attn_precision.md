# Flash Attention kernel produces uncorrelated output vs standard attention

## Summary

The kernel-fusion-based flash attention kernel (`programming_examples/flash_attention/kernel_fusion_based`) produces output with **correlation 0.13-0.34** against CPU F32 standard attention for all tested configurations at `LQ=LK=2048`. The output values are numerically plausible (in the correct range) but mathematically incorrect — the kernel is computing the wrong attention pattern.

The existing `make run` / LIT tests report PASS because they use element-wise tolerance (`atol=0.15, rtol=0.04`) which cannot detect this class of bug. Correlation-based validation reveals the issue.

## Impact

This blocks using NPU flash attention for LLAMA-3.2-1B inference (32 heads, 8 KV heads, seq_len=2048, causal). Currently using CPU `attention_reference()` fallback.

## Reproduction

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Run precision sweep (measures correlation against CPU F32 reference)
python3 test_precision.py

# Expected output:
# Config               corr    max_err  mean_err  4%-fail   out_std
# -----------------------------------------------------------------
# 4h MHA             0.131730   0.7176   0.0727   10.13%   0.1189
# 8h MHA             0.175255   1.2168   0.0758   11.34%   0.1188
# 8h GQA             ...        ...      ...      ...      ...
# LLAMA              0.287155   0.7109   0.0639    6.53%   0.0845
# LLAMA causal       0.343386   2.1602   0.0958   17.91%   0.1191
#
# Expected: corr > 0.99 for correct kernel

# Single config test:
python3 test_precision.py --num-heads 32 --num-kv-heads 8 --causal

# Compare with make run (reports PASS for 4h/8h despite corr=0.13):
make run LQ=2048 LK=2048 LQP=256 LKP=64 DK=64 DV=64 NUM_HEADS=8 NUM_KV_HEADS=8
# PASS! (but corr=0.175)
```

## Analysis

### Correlation vs element-wise tolerance

| Config | corr vs F32 ref | mean_err | `make run` | Why `make run` passes |
|--------|----------------|----------|-----------|----------------------|
| 4h MHA (2048) | **0.132** | 0.073 | PASS | mean_err < atol=0.15 |
| 8h MHA (2048) | **0.175** | 0.076 | PASS | mean_err < atol=0.15 |
| LLAMA (32h/8kv) | **0.287** | 0.064 | FAIL | >2% element mismatches |
| LLAMA causal | **0.343** | 0.096 | FAIL | >2% element mismatches |
| LIT causal (2h, 16K) | **0.243** | 0.041 | PASS | Long seq → tiny variance |

For reference, IRON's MHA kernel achieves **corr=0.998** at the same LLAMA scale (32 heads, 2048 seq, causal).

### Why element-wise tests miss this bug

- Flash attention output is a softmax-weighted average of V values
- With V uniform in [0, 3], output clusters around **1.5** regardless of attention pattern
- Mean absolute error ~0.07 is small relative to `atol=0.15`
- But the **pattern** of which V positions get high attention weight is wrong
- Correlation measures pattern correctness; element-wise tolerance only measures value range

### Root cause hypothesis

The kernel produces output values in the correct range (softmax-weighted averages of V), but the attention weights are wrong. This could be from:
1. Incorrect softmax normalization in the tiled/cascaded implementation
2. Wrong Q/K score computation or scaling
3. Buffer addressing issues in the cascade merge across pipeline stages
4. DMA data movement errors between tiles

## Suggested test improvement

The `run_test` call in `attn.py` should add correlation-based validation:

```python
# Current (misses the bug):
runner.run_test(
    mlir_module,
    inputs=[input_q, input_k, input_v, input_m],
    expected_outputs=[sdpa_output],
    atol=0.15, rtol=0.04, max_mismatch_percentage=2,
)

# Suggested: add correlation check after run_test
# Or use test_precision.py for validation
```

## Environment

- Hardware: AMD Ryzen AI NPU2 (Strix, AIE2P)
- MLIR-AIR: latest (includes PR #1438 fix for causal BD exhaustion)
- Test parameters: LQ=LK=2048, LQP=256, LKP=64, DK=DV=64, val_range=3
- `test_precision.py` measures Pearson correlation between NPU output and CPU F32 standard attention reference

## Files

- `test_precision.py` — Self-contained precision measurement script
- `attn.py` — Flash attention kernel generator and test
- `attn.cc` — C++ kernel implementation
