---
name: evaluate-deployment
description: Independent audit of an LLM deployment (under programming_examples/<model>/) produced by deploy-new-llm. Spawns a fresh subagent that treats the deployment as UNTRUSTED and re-derives every correctness/perf claim from scratch instead of reading the deployment's own LESSONS/progress/perf docs. Invoke as `/evaluate-deployment <model_dir>` to audit one deployment, or auto-spawn from deploy-new-llm Phase 7.
---

## Purpose

The `deploy-new-llm` skill chain is autonomous. Once a deployment is tagged
"complete", the only artifacts of correctness are docs the same agent wrote
(`progress.md`, `LESSONS.md`, `phase{N}_*.md`). This skill provides
**independent verification** by:

1. Spawning a FRESH subagent with no context from the deployment session
2. Telling it to TREAT THE DEPLOYMENT AS UNTRUSTED
3. Having it RE-DERIVE every claim (correctness gates, perf numbers,
   end-to-end output) before reading any deployment-self-reported numbers
4. Comparing measured-vs-claimed and producing a PASS/WARN/FAIL report

The output is a single `<model>/docs/evaluation_report.md`. **The deployment
should NOT be considered trustworthy until this report is generated and
its overall verdict is PASS or PASS-with-warnings.**

## When to use

- **Manual audit**: `/evaluate-deployment <model_dir>` — to verify a
  single deployment finished autonomously
- **Phase 7 auto-spawn**: invoked automatically from `deploy-new-llm` after
  `finalize-deployment` (Phase 6) PASSES, BEFORE the deployment is tagged
- **Cross-deployment regression**: invoke on ALL deployments after any
  shared-infra change (e.g., a `_llm_shared/` or `matvec.py` modification)
  to catch back-compat breaks

## Inputs

`<model_dir>`: path or directory name under `programming_examples/`
(e.g., `qwen25_1_5b`).

## Workflow

Spawn ONE subagent (general-purpose) with the `evaluate_deployment_prompt`
below. The subagent runs the 4 check categories sequentially and writes
the report.

### Critical constraints for the subagent

- **Independence**: do NOT read the following BEFORE forming your own
  measurements: `<model>/docs/development_progress/LESSONS.md`,
  `<model>/docs/development_progress/progress.md`,
  `<model>/docs/development_progress/phase{N}_*.md`. You may CITE them
  AFTER measuring (to compare your numbers vs claimed).
- **Re-derivation**: every PASS/FAIL verdict must be backed by a number
  YOU measured during this evaluation, not a number you read from a doc.
- **Skepticism heuristics** (look for these specifically):
  - Top-1 token gates that pass on trivially-likely tokens (` `, `,`, `.`,
    common stop words). Add 2 adversarial prompts not in the canonical set
    and verify NPU output matches CPU reference there too.
  - NPU prefill timing < 0.1 s for ≥ 16 layers — too fast, kernel may not
    have actually run (silent CPU fallback or stale cached output).
  - "Match rate ≥ 80%" claims where the actual measured value isn't shown.
  - LESSONS.md claims that reference code paths that don't exist (e.g.,
    "we use Option C wrapper" but `cpu_attn=True` in the production runner).
  - Perf numbers reported from a SINGLE run with no trial-to-trial
    variance estimate.
- **Reproducibility**: run `make run` (or equivalent) at least TWICE and
  verify the generated text is identical (or differs only in a documented
  way like temperature sampling).

### Check categories

#### Category 1 — Static audit (cheap, no execution)

1. List files in `<model_dir>` and verify the expected scaffold is present:
   - `<model>_weights.py`, `<model>_reference.py`, `<model>_inference.py`
   - `Makefile` with at minimum: `compile`, `run`, `verify`, `clean` targets
   - `README.md`, `CLAUDE.md`, `TODO.md`
   - `docs/development_progress/` directory
2. Check Makefile targets actually point to existing scripts.
3. Search the code for suspicious patterns (using Grep, NOT just visual scan):
   - `TODO|FIXME|XXX|HACK|TEMPORARY|hardcoded|hard-coded` in non-doc files
   - `return\s+(True|"PASS"|0)\s*$` after a `# TODO|# placeholder` comment
   - `if False:` or commented-out `assert` blocks
   - Mocked imports: `import mock|MagicMock|patch`
4. Cross-check claims in CLAUDE.md / README.md against code:
   - If README claims "uses NPU FA Option C", grep for the wrapper
     installation in the inference script
   - If README claims "X tok/s", note it (you'll re-measure in Cat 5)
5. Output: `[PASS] Static audit: N expected files present, no suspicious patterns`
   or `[WARN]/[FAIL]` with specifics.

#### Category 2 — Weight + reference smoke (~30 s)

1. Run `python3 <model>_weights.py` (or equivalent CLI) — verifies
   safetensors load + shape asserts pass.
2. Run `python3 <model>_reference.py --prompt "The capital of France is" --verify`
   (or equivalent). Verify:
   - Top-1 token from CPU reference is sensible AND matches HF transformers
     within the script's own `--verify` gate.
   - Logits correlation > 0.999 vs HF.
3. Output `[PASS]` if both checks succeed; `[FAIL]` with traceback otherwise.

#### Category 3 — Per-phase re-run (~5–10 min)

Re-run the deployment's own per-phase test scripts AND independently
compute the gates from their output:

1. **Phase 2** (single-block): `python3 <model>_phase2_test.py [args]`.
   Parse the output for `cosine_sim` and `per_pos_min`. Verify they meet
   the head_dim-scaled gate (>0.99 whole; >0.98 for hd=128, >0.99 for
   hd≤64). Note: if measurement < claimed by > 5%, FLAG.
2. **Phase 3** (full model): `python3 <model>_phase3_test.py`.
   Parse the per-prompt table. Verify decisive top-1 = all-pass and
   competitive top-5-overlap = all-pass.
3. **Adversarial Phase 3**: re-run Phase 3 with TWO additional prompts
   NOT in the canonical set (e.g., `"Light travels at"`, `"DNA stands for"`).
   Verify NPU top-1 is in CPU's top-5 for both. Catches over-tuning to the
   canonical prompts.
4. **Phases 4 + 5** (perf): SKIP at v1 (too slow); just check that the
   phase4/5 docs exist and contain timing tables. v2 will re-measure.
5. Output: per-phase PASS/FAIL with measured numbers.

#### Category 4 — End-to-end reproducibility (~1 min)

1. Run `make run N_TOKENS=5` TWICE.
2. Verify:
   - Both runs complete without traceback.
   - Generated text from the two runs is byte-identical (greedy decode
     should be deterministic). If not, FLAG (non-determinism in
     "deterministic" path is a real bug).
   - The first generated token matches the deployment's claimed first
     token (if any is documented in README/phase6_finalize.md).
3. NPU-execution sanity:
   - Check that `prefill_kernel_cache/` and/or `decode_kernel_cache/`
     exist and contain `.elf` artifacts.
   - Check that the second `make run` is meaningfully faster than the
     first (kernel cache hit) — if first vs second timing differs by < 5%,
     either kernels weren't cached, or kernels weren't really running.
4. Output: PASS / WARN / FAIL.

#### Category 5 — Perf integrity (multi-trial, ~5 min) — v2

Single-shot perf claims are fragile (system load, thermal, BO state).
Re-measure with multiple trials and explicit cold/warm protocol; flag if
the claim is outside the measured ± std window.

1. **Cold prefill**: clear `prefill_kernel_cache/` only momentarily —
   actually, DO NOT clear (recompiling burns ~2 min and isn't part of
   "cold prefill"). Instead, define cold = first run after a fresh
   `python` process; warm = subsequent runs in the same process. Use
   `make profile` (the deployment's own perf script) which already does
   this internally — it's the ground truth for the deployment's perf
   measurement window.

2. **Warm prefill, N=5 trials**: re-run the warm-loop 5 times. Compute
   mean, std, median. Compare claim:
   - PASS if claim ∈ [mean - std, mean + std]
   - WARN if claim within ±20% of mean
   - FAIL if claim differs by > 20%
   - ALSO note: if std/mean > 10%, flag the deployment's measurement as
     "high variance — single-shot claim was lucky/unlucky".

3. **Decode tok/s, N=20 tokens steady-state**: re-run; compute median
   ms/token over the last 15 (skip first 5 as warm-up). Same gate as
   prefill.

4. **Anti-fallback heuristics** (measure-then-compare):
   - NPU prefill ms/layer should be > 5 ms (a CPU forward pass at the
     same shape would be much slower, but a "kernel didn't actually run"
     measurement looks suspiciously like 0.01 ms or the kernel cache
     load time).
   - For NPU FA path: `flash_attn.elf` must exist in cache AND the
     measurement should distinguish from CPU-attn — re-run with
     `--cpu-attn` and verify NPU FA path is at least 1.5× faster (real
     speedup) or comparable (if kernel didn't really fire).
   - LM-head GEMV timing should be > 50 ms — if < 10 ms, kernel may not
     have actually run; if > 1 s, the BO preload didn't fire.

5. Output: per-metric PASS/WARN/FAIL with measured mean ± std.

**Skip Category 5 entirely** if the deployment is small (n_layers ≤ 16)
where single-shot variance is naturally low. Phase 5 doesn't apply for
quick smoke audits.

#### Category 6 — Cross-deployment regression (~10–30 min) — v2

If the audit is being run because of a shared-infra change (e.g., a
modification to `_llm_shared/`, `matvec.py`, `llama3/llama3_*.py`, or
`llama3/multi_launch_builder/`), back-compat for OTHER deployments must
be verified. Otherwise this category is N/A.

**Trigger heuristic** (compute first, skip if not triggered):

```bash
# In the repo root, identify shared-infra paths in the recent diff
shared_paths=(
  "programming_examples/_llm_shared/"
  "programming_examples/matrix_vector_multiplication/"
  "programming_examples/llama3/llama3_prefill.py"
  "programming_examples/llama3/llama3_decode.py"
  "programming_examples/llama3/llama3_inference.py"
  "programming_examples/llama3/multi_launch_builder/"
)
git diff main..HEAD --name-only | grep -E "^($(IFS=\|; echo "${shared_paths[*]}"))"
```

If the grep returns any matches, run Category 6. Otherwise mark N/A.

**For each OTHER deployment** (i.e., not the one being primarily
audited): re-run their Phase 2 + Phase 3 test scripts (correctness only;
perf is out of scope here). Verify:
- Phase 2 cosine and per-pos still pass their head_dim-scaled gate
- Phase 3 decisive top-1 + competitive top-5 still pass
- No NaN

If ANY other deployment regresses, FAIL Category 6 with the specific
deployment + measurement. The shared-infra change must be reverted or
fixed before the original deployment can be tagged.

NPU exclusivity: run sequentially — only ONE process can hold the NPU
at a time (XRT enforces /tmp/npu.lock). With 3 OTHER deployments,
budget ~10 min each = 30 min for full cross-deployment regression.

### Report format

Write to `<model_dir>/docs/evaluation_report.md`. Template:

```markdown
# Independent evaluation report — <model_dir>

**Evaluator**: evaluate-deployment skill (subagent: general-purpose)
**Date**: <YYYY-MM-DD>
**Verdict**: PASS / PASS-with-warnings / FAIL

## Measured vs claimed (the headline)

| Metric | Measured (this run) | Claimed (deployment doc) | Verdict |
|---|---|---|---|
| Phase 2 cosine | <measured> | <from phase2_block.md> | ✓/⚠/✗ |
| Phase 3 decisive top-1 | <measured> | <claimed> | |
| Phase 3 competitive top-5 | | | |
| Adversarial top-5 | | (no claim — new check) | |
| End-to-end first token | | | |
| Reproducibility (2 runs identical) | | | |

## Per-category results

### Category 1 — Static audit
[PASS/WARN/FAIL] <details>

### Category 2 — Weight + reference smoke
[...]

### Category 3 — Per-phase re-run
[...]

### Category 4 — End-to-end
[...]

## Issues surfaced

(Numbered list of WARNs and FAILs with specifics; cite file:line.)

## What was NOT checked (v1 scope limitation)

- Perf numbers (Phase 4 prefill warm timing, Phase 5 decode tok/s) — v2.
- Cross-deployment regression — v2.
- Multi-trial perf variance — v2.
```

## Verification (skill gate)

The skill itself does NOT have a PASS/FAIL — its output IS the report.
Treat the report's overall verdict as authoritative for the deployment.

## Failure modes

- Subagent reads LESSONS/progress before measuring → skill instructions
  weren't strict enough; flag as a bug in the skill prompt itself
- Subagent declares PASS without showing measured numbers → reject the
  report; re-spawn with stricter instructions
- A claimed perf number is way off measured → write the discrepancy into
  the report; do NOT fail the deployment for v1 (perf isn't measured yet
  in v1)

## Update protocol

This is the terminal verification phase. The deployment's TODO.md gets a
new entry "Independently evaluated YYYY-MM-DD: <verdict>" and the report
file is referenced from the deployment's progress.md.
