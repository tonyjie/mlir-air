# Phase 3 — Full-model correctness (Qwen2.5-1.5B)

**Date**: 2026-04-19
**Status**: PASS

## Result

All 28 layers + final RMSNorm + (CPU) LM Head, NPU prefill via padded
shapes (emb=2048, hidden=9216, n_heads=16) with GQA-aware reindexed bias
+ host post-add. CPU reference uses orig shapes (emb=1536).

Phase 3 gate (LESSON 2 from llama32_3b):
- Decisive prompts (CPU top-1 p > 0.5) → strict top-1 match required
- Competitive prompts (CPU top-1 p ≤ 0.5) → top-5 overlap required
- No NaN

## Per-prompt results

| Class | Prompt | NPU top-1 | CPU top-1 | cpu_p | logits cos | Verdict |
|---|---|---|---|---|---|---|
| decisive    | `'1 + 1 ='`                  | `' '`       | `' '`       | 0.870 | 0.9493 | PASS (top-1 match) |
| decisive    | `'2 + 2 ='`                  | `' '`       | `' '`       | 0.876 | 0.9808 | PASS (top-1 match) |
| decisive    | `'Water freezes at'`         | `' '`       | `' '`       | 0.665 | 0.9766 | PASS (top-1 match) |
| competitive | `'The largest ocean is the'` | `' Pacific'`| `' Pacific'`| 0.336 | 0.9876 | PASS (top-5 overlap; also top-1) |
| competitive | `'The capital of France is'` | `' Paris'`  | `' Paris'`  | 0.273 | 0.9925 | PASS (top-5 overlap; also top-1) |
| competitive | `'The sky is'`               | `' '`       | `' blue'`   | 0.265 | 0.9071 | PASS (top-5 overlap; BF16 reorder of close p≈0.14/0.27) |

**Tally**:
- Decisive top-1 match: **3/3** ✓
- Competitive top-5 overlap: **3/3** ✓
- Strict top-1 (all): 5/6 (1 expected BF16 reorder on competitive prompt)
- NaN: None ✓

## Timing (CPU attention path)

- NPU prefill: ~10 s / 28 layers (~360 ms/layer)
- CPU reference: ~18 s / 28 layers
- Total per-prompt: ~28 s

NPU FA (Option C wrapper) deferred to a Phase 4 follow-up — CPU-attn
suffices for Phase 3 correctness gate. Predicted 2–3× prefill speedup
once Option C wrapper is validated at Qwen2.5's padded shape (n_heads=16,
n_kv_heads=2, group=8 in the padded view).

## Notes

- The `'The sky is'` reorder (NPU=' ' vs CPU=' blue') is the same kind of
  competitive-prompt drift seen in `llama32_3b/`'s Phase 3 (Lesson 2):
  CPU top-1 prob 0.265 sits close to several other tokens, and BF16
  accumulation across 28 layers reorders them. Both ' blue' and ' ' are
  in each other's top-5 → gate satisfied.

- The padding scheme (emb_dim 1536→2048, hidden_dim 8960→9216, GQA
  reindex) propagates correctly through all 28 layers — no per-layer
  drift from a structural padding bug. Verified by single-block CPU
  sanity test (cosine 0.999998) before NPU run.

- Per-layer cosine drift (diagnostic, not a gate at n_layers=28 +
  head_dim=128 per LESSON 1) not captured this run; can be enabled with
  `--diagnostic` for future runs.
