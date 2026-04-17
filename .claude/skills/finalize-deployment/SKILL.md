---
name: finalize-deployment
description: Phase 6 of LLM deployment — produce final perf report, update knowledge base with new lessons, harvest any reusable patterns into _llm_shared/. Invoked after Phase 5 gate.
---

## Purpose
Close the deployment cleanly: write the perf summary, document lessons (especially novel failure modes encountered), and identify any pattern that's now used by 2+ models and should be promoted to shared infrastructure.

## Knowledge base references
- `programming_examples/llama3/docs/development_progress/progress.md` — reference final-state document

## Workflow

### Step 1: Write perf summary
In `<model>/docs/development_progress/progress.md` (or its own file), produce a table:

```
| Phase | Outcome | Key metric |
|-------|---------|------------|
| Bootstrap | PASS | Reference top-1 stable |
| Per-kernel shapes | PASS | N/N kernels validated |
| Single block | PASS | corr=X.XX, mae=Y.YY |
| Full model | PASS | top-1 match 3/3 prompts |
| Prefill perf | PASS | Z ms (5/5 patterns applied) |
| Decode perf | PASS | W ms/token (4/5 patterns applied) |
```

Include comparison vs CPU reference and (if available) IRON baseline.

### Step 2: Update LESSONS.md
For each escalation that became a new recipe in this deployment:
- Append to `<model>/docs/development_progress/LESSONS.md`
- Cross-reference to the new cross-cutting skill (if one was authored)

### Step 3: Harvest reusable patterns
Audit the deployment for things that look generic:
- A new helper function that ended up identical to an existing one in `_llm_shared/kernel_builder/` → consider promoting
- A new debug recipe used twice (in this deployment AND a known LLAMA case) → consider promoting from `<model>/` notes to a new cross-cutting skill

Don't promote speculatively — only if there's evidence of 2+ uses.

### Step 4: Tag the deployment
```bash
cd <repo_root>
git tag -a deployment-<model_name>-v1 -m "First validated NPU2 deployment of <model_name>"
```

## Verification (Phase 6 gate)

Phase 6 PASSES when:
- `progress.md` complete with summary table
- `LESSONS.md` reflects any novel failures
- Perf comparison table includes CPU reference numbers (and IRON if available)

## Update protocol
This is the terminal phase. Set `TODO.md` Phase 6 to PASSED. Deployment is complete.
