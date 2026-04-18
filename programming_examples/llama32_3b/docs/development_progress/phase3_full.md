# Phase 3 — Full-Model Correctness

**Date**: 2026-04-18
**Status**: PASS with adapted gate (decisive vs competitive prompts).
**Strict-3/3 gate**: 2/3 of original canonical prompts match top-1 (1 is a top-2 reorder).
**Adapted gate (LESSONS Lesson 2)**: 4/4 decisive prompts top-1 match exactly + 2/2 competitive prompts have perfect top-5 overlap + no NaN. **PASS**.

## Configuration
- 28 layers wired through `run_transformer_block`
- CPU attention fallback (NPU FA at head_dim=128 deferred to Phase 4)
- Final RMSNorm + LM Head on CPU (Phase 5 will move LM Head to NPU)
- seq_len=2048, BF16 weights, BF16 GEMM outputs

## Top-1 results

### Original 3 canonical prompts (skill spec)

| # | Prompt | class | NPU top-1 (prob) | CPU top-1 (prob) | top-1 match | logits cos |
|---|---|---|---|---|---|---|
| 1 | `'The capital of France is'` | competitive | `' the'` (0.113) | `' Paris'` (0.246) | NO (top-5 reorder) | 0.984 |
| 2 | `'1 + 1 ='`                    | decisive    | `' '` (0.615)    | `' '` (0.737)     | **YES** | 0.990 |
| 3 | `'The sky is'`                 | competitive | `' the'` (0.340) | `' the'` (0.331)  | **YES** | 0.984 |

### Added decisive prompts (CPU top-1 prob > 0.5 → above BF16 reorder noise)

| # | Prompt | class | NPU top-1 (prob) | CPU top-1 (prob) | top-1 match | logits cos |
|---|---|---|---|---|---|---|
| 4 | `'2 + 2 ='`                    | decisive | `' '` (0.334)         | `' '` (0.529)         | **YES** | 0.943 |
| 5 | `'Water freezes at'`           | decisive | `' '` (0.511)         | `' '` (0.707)         | **YES** | 0.955 |
| 6 | `'The largest ocean is the'`   | decisive | `' Pacific'` (0.658)  | `' Pacific'` (0.816)  | **YES** | 0.978 |

### Adapted-gate result

- Decisive prompts (CPU p > 0.5) top-1 match: **4/4** ✓
- Competitive prompts top-5 overlap (CPU top-1 ∈ NPU top-5 AND NPU top-1 ∈ CPU top-5): **2/2** ✓
- No NaN ✓
- Strict skill gate (all top-1 match): 5/6 — fails on prompt 1 only.

## Top-5 overlap analysis (prompt 1, the failing case)

Same 5 tokens, just the top-2 are reordered. Probabilities are extremely close — well within BF16 production noise:

| rank | NPU | CPU |
|---|---|---|
| 1 | `' the'` (p=0.113) | `' Paris'` (p=0.246) |
| 2 | `' Paris'` (p=0.094) | `' the'` (p=0.136) |
| 3 | `' a'` (p=0.088) | `' a'` (p=0.095) |
| 4 | `' located'` (p=0.072) | `' located'` (p=0.074) |
| 5 | `' also'` (p=0.052) | `' also'` (p=0.049) |

For all three prompts: **CPU top-1 ∈ NPU top-5 AND NPU top-1 ∈ CPU top-5**.

## Per-layer cosine drift (prompt 1, --diagnostic mode)

Monotonic accumulation, no single-layer drop. Drift below the 0.95 gate starts at layer 24:

```
Layer  0: 0.997103  / 0.989294
Layer  4: 0.996149  / 0.982139
Layer  8: 0.993734  / 0.974524
Layer 12: 0.992135  / 0.971093
Layer 16: 0.986232  / 0.963122
Layer 20: 0.969838  / 0.933568
Layer 23: 0.950547  / 0.901683
Layer 24: 0.942425  / 0.891431  <-- below 0.95
Layer 25: 0.932266  / 0.884307
Layer 26: 0.916890  / 0.876216
Layer 27: 0.881133  / 0.843667
```

Final cosine: 0.881 (whole) / 0.844 (per-pos min). No NaN.

## Why this is not a kernel bug

1. **Phase 2 single-block**: cosine 0.999 / per-pos 0.989 with MAE 0.005 (5× lower than smollm2). All kernels independently correct to BF16 precision.
2. **Drift is monotonic across layers** — no single-layer cliff. Inconsistent with a layer-indexed weight bug or a bad kernel.
3. **Top-5 sets are identical** on the failing prompt — only the top-2 are swapped, and their probabilities are very close (0.246/0.136 in CPU = 1.8× ratio, easily flipped by BF16).
4. **Skill failure-mode #3**: "Drift starts at layer 0 and worsens → likely BF16 accumulator issue throughout; F32 accumulator fix per LLAMA Phase 3" — exactly our pattern.

The root cause is **BF16 production accumulation noise**, scaled by:
- head_dim 64→128 (2× more attention-output noise per row)
- emb_dim 2048→3072 (1.5× more matmul-output noise per row)
- n_layers 16→28 (1.75× more layers to compound across)

Combined ~5× more accumulated drift than llama3-1B (per LESSONS Lesson 1).

## Llama3 historical precedent (from llama3/docs/development_progress/progress.md)

llama3 hit the same kind of issue at 16 layers:
- 2026-03-13: "16-layer top-1 still incorrect" with BF16-output GEMMs
- 2026-03-13: F32-output GEMM fix raised single-step corr 0.976 → 0.9999 (52× better)
- 2026-03-19: Upstream rounding fix landed; reverted to BF16-output (smaller, faster)
- 2026-03-16 PASS verdict: NPU said `' Paris'`, CPU said `' the'` — both "valid", disagreement
  attributed to "benign BF16 numerical noise". 16 layers, logits corr 0.972.

We're the inverse: NPU `' the'`, CPU `' Paris'`. Same kind of BF16 reorder; we have 28 layers
(1.75× deeper), wider GEMMs (1.5×), wider heads (2×) — so the noise budget is larger.

## Decision options

### Option A — accept top-5-overlap PASS (no code change)
**Pros**: zero engineering risk, 2/3 prompts match top-1 exactly, top-5 sets perfectly overlap on the
failing case, semantically equivalent to llama3's accepted state.
**Cons**: skill's strict gate is "3/3 top-1 match" — we'd be relaxing that to "2/3 top-1 OR top-5
overlap on the failure". This is captured as an honest finding rather than a gate-bypass.

### Option B — F32-output Down GEMM
Modify `o_ffn_multi.py` (or the Down-GEMM portion of it) to produce F32, then cast to BF16 after
the residual add. The Down GEMM has K=8192 (largest accumulation depth) and is the dominant noise
source per llama3 history.
**Pros**: targeted fix at the dominant noise contributor; smaller refactor than full F32-output
everywhere.
**Cons**: edits a shared file (`programming_examples/llama3/multi_launch_builder/o_ffn_multi.py`)
that llama3 also depends on; risks regressing llama3 perf or breaking its multi-launch ELF
intermediates. Estimated 2–4 hours of work + revalidate llama3 + revalidate llama32_3b.

### Option C — F32-output for all 7 GEMMs
The historical 2026-03-13 fix. Major refactor of multi-launch ELFs to carry F32 intermediates.
**Pros**: rigorous, would push per-layer cos to 0.999+ uniformly.
**Cons**: ~1–2 days of work; reverts the perf optimization that llama3 banked in
2026-03-19.

### Option D — narrow scope to "factually correct continuations"
Pick prompts where top-1 is decisive (CPU prob > 0.5). Example: `'1 + 1 ='` already PASSES because
CPU top-1 has p=0.74. The current prompt 1 fails specifically because CPU top-1 only has p=0.246
(competitive with second place). A revised canonical prompt set (e.g., `'2 + 2 ='`, `'The largest
ocean is the'`, `'Water freezes at'`) where CPU top-1 has p > 0.5 would likely 3/3 match.
**Pros**: tests on prompts that are decisive in the model's distribution; lower noise-sensitivity.
**Cons**: feels like cherry-picking; doesn't address the underlying drift.

## Resolution (2026-04-18)

User chose **Option A + D**: accept top-5-overlap on competitive prompts + add 3 decisive prompts to
the canonical set. The expanded test produces 4/4 decisive top-1 match + 2/2 competitive top-5
overlap + no NaN. Captured as `LESSONS.md` Lesson 2.

If Phase 4 NPU FA work surfaces a chance to also do an F32 Down-GEMM pass at the same time, fold
it in then. Otherwise, defer the F32-output investment.

## Items surfaced to later phases

| Item | When | Severity |
|---|---|---|
| **F32-output Down GEMM**: would push per-layer cos to 0.999+ uniformly and likely move competitive prompts to top-1 match. ~2–4 hour refactor of `o_ffn_multi.py` (shared with llama3 — must revalidate llama3 after change). Not on critical path; only do if a downstream metric demands it. | Phase 4/6 (deferred) | low — accuracy is acceptable |
| **`validate-full-model-correctness` skill update**: add the "decisive vs competitive prompt" gate distinction (CPU top-1 p > 0.5 → strict top-1 match; p ≤ 0.5 → top-5 overlap). Captures the BF16 reorder reality at depth ≥ ~24 layers without weakening the gate where the model is decisive. | Phase 6 (skill update) | low — captured as Lesson 2 |
