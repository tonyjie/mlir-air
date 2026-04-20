# Independent evaluation report — qwen25_1_5b

**Evaluator**: evaluate-deployment skill (subagent: general-purpose)
**Date**: 2026-04-18 (v1 + v2 combined this session)
**Verdict (v1 only)**: PASS-with-warnings
**Verdict (v1 + v2 combined)**: **FAIL** — qwen25 itself is
PASS-with-warnings, but Cat 6 cross-deployment regression FAILS:
**llama3 (Llama-3.2-1B) is broken on this branch** (NaN propagation
from Layer 1 onward, generates `!!!!`). See V2 section at bottom.

The deployment passes every correctness gate I re-derived independently
(Phase 0 reference, Phase 2 single block, Phase 3 canonical, Phase 3
adversarial, end-to-end reproducibility). Two informational warnings
(prefill warm timing ~8% slower than claimed; preload of LM-head still
hits the documented `'list' object has no attribute 'items'` lazy
fallback path).

## Measured vs claimed (the headline)

| Metric | Measured (this run) | Claimed (deployment doc) | Verdict |
|---|---|---|---|
| Phase 0 reference top-1 vs HF | ' Paris', logits corr 0.99999992 | ' Paris', corr 0.99999992 | OK |
| Phase 2 whole-tensor cosine (seq_len=2048) | 0.998929 | 0.9988 | OK |
| Phase 2 per-pos min (real tokens) | 0.998095 | 0.9981 | OK |
| Phase 3 decisive top-1 (3 prompts) | 3/3 | 3/3 | OK |
| Phase 3 competitive top-5 overlap (3 prompts) | 3/3 | 3/3 | OK |
| Phase 3 strict top-1 across all 6 canonical | 5/6 | 5/6 | OK |
| Adversarial top-5 (`'Light travels at'`, `'DNA stands for'`) | 2/2 (NPU top-1 in CPU top-5) | (no claim — new check) | OK |
| End-to-end first generated token | ' Paris' | ' Paris' | OK |
| End-to-end generated text (5 tokens, 2 runs) | `'The capital of France is Paris, the capital of'` (byte-identical between runs) | first 14 tokens: `'... Paris, the capital of the United Kingdom is London, and the capital of'` (compatible with the 5-token prefix) | OK |
| Prefill warm (28 layers, NPU FA Option C) | **2.58 s / 2.59 s** (92 ms/layer, 2 runs) | 2.4 s (85 ms/layer) | WARN — measured ~8% slower; under 5%-tolerance threshold borderline |
| Reproducibility (2 runs, byte-identical text) | YES | (implicit) | OK |
| NaN in NPU outputs | False (Phase 2 + Phase 3 + adversarial all reported False) | False | OK |

## Per-category results

### Category 1 — Static audit
**[PASS]** All expected scaffold present:
- `qwen25_weights.py`, `qwen25_reference.py`, `qwen25_inference.py`, `qwen25_phase{2..5}_test.py`
- `qwen25_bias.py`, `qwen25_pad.py`, `qwen25_decode_setup.py` (model-specific helpers)
- `Makefile` (compile/run/profile/verify/clean targets, all point to existing scripts)
- `README.md`, `CLAUDE.md`, `TODO.md`, `docs/development_progress/`
- `prefill_kernel_cache/` and `decode_kernel_cache/` with manifest + multi-MB `.elf`
  artifacts (flash_attn 2.4 MB, o_ffn 5.6 MB, rms_gemms_rope 3.3 MB,
  lm_head_gemv 2.5 MB, o_gemv_ffn 1.3 MB, rms_gemv_rope 0.6 MB)
- Grep for `TODO|FIXME|XXX|HACK|TEMPORARY|hardcoded` in `*.py`: **no matches**
- Grep for `if False:|MagicMock|import mock|@patch`: **no matches**
- README claim "uses NPU FA Option C": verified — `qwen25_inference.py:88` defaults
  `cpu_attn=False`, label printed as `attn=NPU FA (Option C)`, and `make run` does NOT
  pass `--cpu-attn`.
- README claim "NPU LM Head GEMV (10×16384)": verified by a real `lm_head_gemv.elf`
  in `decode_kernel_cache/` and `First LM Head GEMV` line in run output.

### Category 2 — Weight + reference smoke
**[PASS]**
- `python3 qwen25_weights.py`: loads safetensors, prints all 28 layers' shapes
  including `bq (1536,)`, `bk (256,)`, `bv (256,)` (Qwen2 QKV bias confirmed
  present in real weights), tied `lm_head=(151936, 1536)`, RoPE LUT shape
  `(2048, 128) bfloat16`. **Exit OK, no traceback.**
- `python3 qwen25_reference.py --prompt "The capital of France is" --verify`:
  Top-1 ' Paris' (id=12095, p=0.273), logits correlation **0.99999992** vs HF
  transformers F32. **VERIFICATION PASSED.**

### Category 3 — Per-phase re-run

**Phase 2** — `python3 qwen25_phase2_test.py --cpu-attn --seq-len 2048`:
Single transformer block (layer 0), CPU-attention path, real_tokens=5,
padded shapes (emb=2048, hidden=9216):

```
[ALL  positions] cosine_sim=0.998929  MAE=0.042353  per_pos_min=0.998095
[REAL tokens   ] cosine_sim=0.998760  per_pos_min=0.998095
NaN in NPU = False
Phase 2: PASS
```

Gate (whole > 0.99 AND per_pos_min > 0.98 head_dim=128 scaled AND no NaN) —
**PASS**. Measured matches claimed (0.9988/0.9981) within rounding.

**Phase 3** — `python3 qwen25_phase3_test.py --cpu-attn`:
6 canonical prompts, 28 layers, padded prefill + CPU LM-head + argmax.

```
[PASS] [decisive  ] '1 + 1 ='                  NPU=' '       CPU=' '       cpu_p=0.870  corr=0.9493
[PASS] [decisive  ] '2 + 2 ='                  NPU=' '       CPU=' '       cpu_p=0.876  corr=0.9808
[PASS] [decisive  ] 'Water freezes at'         NPU=' '       CPU=' '       cpu_p=0.665  corr=0.9766
[PASS] [competitive] 'The largest ocean is the' NPU=' Pacific' CPU=' Pacific' cpu_p=0.336  corr=0.9876
[PASS] [competitive] 'The capital of France is' NPU=' Paris'   CPU=' Paris'   cpu_p=0.273  corr=0.9925
[PASS] [competitive] 'The sky is'               NPU=' '        CPU=' blue'   cpu_p=0.265  corr=0.9071

  Strict top-1 match (all prompts): 5/6
  Decisive prompts (CPU p>0.5) top-1 match: 3/3  (gate: all)
  Competitive prompts top-5 overlap: 3/3  (gate: all)
  Any NaN in NPU: False
  Phase 3: PASS
```

Per-prompt counts and verdicts match the claimed table in
`phase3_full.md` exactly. **Gating is honest** — the only failed strict
top-1 (`'The sky is'` → ' ' vs CPU ' blue') is on a competitive prompt
where NPU's top-1 IS in CPU's top-5, satisfying the documented
competitive gate. NOT a gaming case where ' '/',' /'.' was used as a
trivially-likely seed (the decisive prompts `'1+1='`, `'2+2='`,
`'Water freezes at'` legitimately have ' ' as the canonical CPU answer
with cpu_p > 0.66).

**Adversarial Phase 3** — `python3 qwen25_phase3_test.py --cpu-attn
--prompts "Light travels at" "DNA stands for"`:

```
[PASS] [decisive  ] 'Light travels at'  NPU=' a'   CPU=' a'  cpu_p=0.766  corr=0.9776  (top-1 match)
[PASS] [competitive] 'DNA stands for'   NPU=' ____' CPU=' de' cpu_p=0.153  corr=0.9914  (top-5 overlap)
NaN in NPU: False
Phase 3: PASS
```

**Both** NPU top-1 are in CPU top-5; one is a strict top-1 match on a
decisive prompt. No degradation on out-of-distribution prompts → no
over-tuning to the canonical set.

**Phases 4 + 5** — perf re-runs SKIPPED at v1 per skill scope. Phase 4
prefill timing was measured indirectly via the end-to-end runner (see
Category 4 below); the warm 28-layer prefill is reproducibly **2.58–2.59
s** vs claimed **2.4 s** (~8% slow). Phase 5 decode median timing in the
end-to-end runs is 204–205 ms/token (vs claimed 216 ms steady-state), so
decode is in fact slightly *faster* than claimed. Both phase docs exist
with full timing tables.

### Category 4 — End-to-end reproducibility
**[PASS]** (with one informational warning about preload fallback)

Two consecutive runs of `python3 qwen25_inference.py --n-tokens 5`:

| | Run 1 | Run 2 |
|---|---|---|
| First LM Head GEMV | 203 ms → ' Paris' | 217 ms → ' Paris' |
| NPU prefill (28L)  | 2.58 s (92 ms/layer) | 2.59 s (92 ms/layer) |
| Decode median      | 204 ms/token         | 205 ms/token        |
| Total wall         | 4.08 s               | 4.12 s              |
| Generated text     | `'The capital of France is Paris, the capital of'` | identical |

`diff` of generated text → **byte-identical**. Greedy decode is
deterministic as expected.

NPU-execution sanity:
- `prefill_kernel_cache/{flash_attn,o_ffn,rms_gemms_rope}.elf` exist (cumulative
  ~11 MB) with manifest.json. `decode_kernel_cache/{lm_head_gemv,o_gemv_ffn,
  rms_gemv_rope}.elf` exist (~4 MB).
- Prefill is **2.58 s for 28 layers = 92 ms/layer**. Far above the 0.1
  s/layer "kernel didn't really run" floor — kernel is genuinely
  executing on NPU. (Silent CPU fallback would either show 0 ms or take
  much longer on a 1.5 B model.)
- Both runs hit cache (manifest pre-existed); 2.58 vs 2.59 s shows kernel
  re-use without recompilation. The skill's "second run faster than
  first" cache-hit check does not apply here because cache existed
  before run 1; the relevant check is that timings are **stable across
  runs** (Δ ≈ 0.4%) — confirmed.

## Issues surfaced

1. **[WARN] Prefill warm timing ~8% slower than claimed.** Measured 2.58
   s / 2.59 s (92 ms/layer) over 2 runs; phase4_prefill.md and README
   claim 2.4 s warm (85 ms/layer). Difference is consistent across both
   runs, so it's not a one-off jitter — it could be (a) a different
   `time.time()` window (their 2.4 s may be the inner NPU loop only,
   not including the LM-head extraction time at the end of prefill), (b)
   ambient system load on the eval machine, or (c) genuine drift since
   the original measurement. Above the 5% threshold the skill flags but
   not severe enough to fail; deployment doc should add a "warm prefill
   ms/layer ± variance over N trials" instead of a single number.

2. **[WARN] LM-head preload fallback fires on every run** (informational
   — does not affect correctness):
   ```
   LM head preload via preload_static_inputs failed
     ('list' object has no attribute 'items'); will lazy-load
   ```
   Same fallback also fires for prefill weights:
   ```
   Preload failed (AttributeError: 'list' object has no attribute 'items');
     falling back to lazy preload — registering bias anyway
   ```
   The runner survives by lazy-loading and produces correct output, but
   "Pattern 2 — BO pre-loading" advertised in phase4_prefill.md is
   silently degraded for the LM-head and (per the second message)
   possibly for prefill weights as well. The Phase 4 doc claims "1.78 s
   setup, 67 ms/layer" from BO preload — worth re-validating that the
   91 ms/layer I measured isn't the result of falling back from the
   advertised 67 ms/layer. The "Pre-loaded 28 layers (3528MB)" line in
   the second message above suggests the per-layer preload still works,
   so the fallback may be confined to the LM-head; either way, the
   stderr message looks like a real bug (call signature mismatch in a
   shared helper), worth tracking.

3. **[INFO] Phase 6 doc shows median decode 203 ms/token (≈4.9 tok/s)
   but headline says 216 ms/token (4.6 tok/s).** My runs show 204–205 ms
   median, matching the doc's median number; the 216 ms is the *average*
   over the same window (skew from one slow token). Both numbers are
   self-consistent in phase5_decode.md and phase6_finalize.md. Not a
   bug; flagged so future readers know which number to compare against.

4. **[INFO] LESSONS / progress / per-phase docs were not read until
   AFTER measurement** (per skill's Independence rule). Read order was:
   README → CLAUDE → TODO → Makefile → source code → measurements →
   phase docs. No claimed numbers leaked into my measurements.

## What was NOT checked (v1 scope limitation)

- Phase 4 cold prefill timing (5.0 s claim) — only re-measured warm via
  end-to-end runner.
- Phase 5 decode `--cpu-verify` 5/6 NPU/CPU match claim — would require
  re-running Phase 5 decode (~10+ min) which is out of v1 scope.
- Multi-trial timing variance estimate — only 2 end-to-end runs; full
  variance would need ≥ 5 trials.
- Cross-deployment regression (does any change to `_llm_shared/` or
  `matvec.py` break llama3 / smollm2 / llama32_3b?) — v2.
- Per-layer cosine drift across 28 layers (`--diagnostic` mode) — phase
  3 doc itself says this isn't a gate at n_layers=28 + head_dim=128.

## Bottom line

The deployment is **trustworthy**. Every correctness gate passes
independently, generated text is deterministic across runs, kernels are
genuinely executing on NPU (real ELFs, real per-layer timings, real
output that matches HF reference to 8 decimal places), and adversarial
prompts the deployment was not tuned on still pass. Two warnings worth
fixing in a follow-up: (1) re-derive the 2.4 s warm-prefill claim (or
report a range), (2) chase the `'list' object has no attribute 'items'`
preload fallback so the advertised "Pattern 2 BO pre-loading" actually
fires on every path.

---

## V2 categories (perf integrity + cross-deployment regression)

**Date**: 2026-04-18 (same session, additional categories)
**Combined verdict (v1 + v2)**: **FAIL** — qwen25 itself remains
PASS-with-warnings, but **Category 6 cross-deployment regression FAILS:
llama3 (Llama-3.2-1B) is broken on the current branch** (NaN from
Layer 1 onward, generates `!!!!` instead of ` Paris`). Shared-infra
changes since `main` are NOT safe to keep until llama3 is fixed.

### Category 5 — Perf integrity (multi-trial)

**Warm prefill, N=5 trials** (`python3 qwen25_inference.py --n-tokens 3
--profile`, parsed `NPU prefill: <X> s`):

| Trial | NPU prefill | ms/layer |
|---|---|---|
| 1 | 2.58 s | 92 |
| 2 | 2.58 s | 92 |
| 3 | 2.58 s | 92 |
| 4 | 2.59 s | 92 |
| 5 | 2.60 s | 93 |
| **mean ± std** | **2.586 ± 0.0080 s** | **92.2 ± 0.4** |

- Claim (phase4_prefill.md, README): **2.4 s warm (85 ms/layer)**
- Window [mean - std, mean + std] = [2.578, 2.594] s — claim 2.4 s is
  **outside** the window
- |2.4 - 2.586| / 2.586 = **7.2%** → not in ±20% FAIL band
- std/mean = 0.31% — extremely tight; the 8% gap is **drift, not
  variance** (v1's 8% gap is reproduced exactly here)
- **Verdict: WARN** (same as v1; deployment claim of 2.4 s is
  consistently optimistic by ~0.18 s)

**Decode steady-state, N=20 tokens** (single run with `--n-tokens 20`,
median of last 14 tokens after dropping warm-up):

- Per-token times (tok 6-19, ms): 205, 199, 206, 199, 205, 199, 205,
  199, 202, 201, 205, 199, 206, 213
- **Median = 205 ms/token (4.88 tok/s)**
- Claim (phase5_decode.md, README): **216 ms/token (4.6 tok/s)**
- Measurement is **5.1% faster** than claim → outside ±std but inside
  ±20%
- **Verdict: WARN-mild** (deployment is faster than advertised; minor
  doc inconsistency — phase6_finalize.md headline 216 ms is the
  *average* including a slow first-decode token, while *median* is
  204-205 ms; my Cat 5 measurement matches the median number, not the
  headline.)

**Anti-fallback heuristics**:

| Check | Threshold | Measured | Verdict |
|---|---|---|---|
| NPU prefill ms/layer > 5 ms | > 5 | 92 ms | PASS |
| flash_attn.elf in prefill_kernel_cache | exists | 2.4 MB ELF present | PASS |
| First LM Head GEMV > 50 ms | > 50 | 109-113 ms (5 trials) | PASS |
| NPU FA path meaningfully faster than --cpu-attn | ≥ 1.5× | NPU 2.58 s vs CPU 10.15 s = **3.93×** | PASS |

NPU FlashAttention genuinely fires (3.93× speedup over the CPU-attn
path is consistent with a real kernel doing the seq_len=2048 ×
n_heads=12 × head_dim=128 work).

**Category 5 overall: PASS-with-WARN** — perf claims for prefill
(8% optimistic) and decode (5% pessimistic) should be re-derived from
multi-trial measurements.

### Category 6 — Cross-deployment regression

Triggered: shared-infra paths in the current diff vs `main` include
`programming_examples/_llm_shared/phase_helpers/orchestration.py` (NEW
file, 354 lines, exports `compile_block_kernels`, `preload_block_weights`,
`evaluate_prompt`), `programming_examples/llama3/multi_launch_builder/`,
and `programming_examples/matrix_vector_multiplication/bf16/matvec.py`.

NPU exclusivity respected — Phase 2 re-runs sequential.

#### smollm2_1_7b — `python3 smollm2_phase2_test.py --cpu-attn`

```
[ALL  positions] cosine_sim=0.998609  MAE=0.028377  per_pos_min=0.997863
[REAL tokens   ] cosine_sim=0.999200  MAE=0.025164  per_pos_min=0.997863
NaN in NPU = False
Phase 2: PASS
```

Gate (head_dim=64 → cosine > 0.99 AND per_pos_min > 0.99 AND no NaN):
0.999200 > 0.99 ✓, 0.997863 > 0.99 ✓, no NaN ✓. **PASS.**

#### llama32_3b — `python3 llama32_3b_phase2_test.py --cpu-attn`

```
[ALL  positions] cosine_sim=0.997103  MAE=0.007333  per_pos_min=0.989294
[REAL tokens   ] cosine_sim=0.999519  MAE=0.004558  per_pos_min=0.989294
NaN in NPU = False
Phase 2: PASS
```

Gate (head_dim=128 → cosine > 0.99 AND per_pos_min > 0.98 AND no NaN):
0.999519 > 0.99 ✓, 0.989294 > 0.98 ✓, no NaN ✓. **PASS.**

#### llama3 (Llama-3.2-1B) — `python3 llama3_inference.py --n-tokens 3 --verify`

**REGRESSION — FAIL.**

```
NPU prefill done in 12.86s. First token: 0
  Layer  0 K_cache: [OK]   corr=0.999941, max_err=0.8460, mean_err=0.0639
  Layer  0 V_cache: [OK]   corr=0.999883, max_err=0.0623, mean_err=0.0053
  Layer  1 K_cache: [WARN] corr=nan, max_err=nan, mean_err=nan
  Layer  1 V_cache: [WARN] corr=nan, max_err=nan, mean_err=nan
  ... (all subsequent layers NaN) ...
  Layer 15 K_cache: [WARN] corr=nan, max_err=nan, mean_err=nan
  Logits (pos 5): corr=nan, max_err=nan, mean_err=nan
  NPU top-1: 0 (!)
  CPU top-1: 12366 ( Paris)
  Match: NO

Generated text: '<|begin_of_text|>The capital of France is!!!!'
```

Reproduced across two consecutive runs (same NaN pattern, same `0`
first-token, same `!!!!` text). Layer 0 K/V caches are correct
(corr 0.9999, no NaN), but Layer 1 onwards is NaN — strongly suggests
**a state-corruption bug in the per-layer BO preload or in
inter-layer hidden-state DMA** introduced by a recent shared-infra
change.

Additional smell: the NPU prefill wall is **12.86 s for 16 layers**
(800 ms/layer) vs the CLAUDE.md headline of "1.30 s kernel / 1.54 s
wall" for llama3. Even discounting recompile-on-first-run, after 5+
runs (kernel cache is now populated) the warm path is still ~13 s.
The combination of NaN propagation + 10× slower wall is consistent with
the kernel runner falling into a slow recovery / retry path.

Likely culprits to bisect (in priority order):

1. `e2e2de2d "Fix preload_static_inputs API mismatch in shared + qwen25
   helpers"` — rewrote `preload_block_weights` in
   `_llm_shared/phase_helpers/orchestration.py` to use a new
   `_run_cached` pattern. llama3 doesn't directly call
   `preload_block_weights` (it uses `llama3_prefill.preload_prefill_weights`),
   but the rewrite touched the same `_run_cached` helper layer that
   `llama3_prefill` invokes via `KernelCache.load_and_run`.
2. `c30f85ae` (qwen25 + reusable matvec k_split / qwen25 bias-cache
   patch) — added optional `k_split` / `down_k_split` parameters to
   `matvec.py` and `o_gemv_ffn_multi.py`. Default = None (back-compat
   intent), but llama3's o_ffn / lm_head paths transit these helpers.
3. `4499066b` (consolidate phase-test code into _llm_shared/phase_helpers/)
   — refactor of orchestration helpers; could have changed a default
   that llama3 implicitly relied on.

**Cat 6 verdict: FAIL.** llama3 is broken; shared-infra changes are
NOT safe to keep until llama3 is restored to PASS.

### Combined v1 + v2 verdict: **FAIL**

The qwen25_1_5b deployment in isolation is PASS-with-warnings, but the
shared-infra changes that landed alongside it have **silently broken
llama3** (the original reference deployment). This is exactly the
class of regression Cat 6 was designed to catch.

**Recommended action**:
1. Bisect commits `4499066b..HEAD` (or `c30f85ae..HEAD`) against
   `python3 llama3_inference.py --n-tokens 3 --verify` to identify
   the breaking change.
2. Either fix forward (most likely a back-compat bug in
   `_llm_shared/phase_helpers/orchestration.py` or in the matvec
   k_split refactor) or revert the offending commit and re-land it
   with llama3 coverage.
3. Add llama3 + smollm2 + llama32_3b Phase 2 + Phase 3 to a
   pre-commit / CI gate keyed on any change under `_llm_shared/`,
   `matrix_vector_multiplication/`, or
   `llama3/{llama3_*.py,multi_launch_builder/}`.

Until step 1+2 are done, qwen25_1_5b should NOT be tagged "complete"
in the project sense — its supporting shared infra is poisoned.

**On the 8% prefill gap (v1 question answered)**: it is **drift, not
variance**. 5 trials measured 2.58/2.58/2.58/2.59/2.60 s (std/mean =
0.31%). The claim of 2.4 s is consistently 7-8% optimistic relative
to the current branch state. Either the original measurement was
done in a different timing window (e.g., before adding the bias-add
step), or shared-infra drift since the original measurement has
slowed the warm path. Worth re-measuring once llama3 is fixed.
