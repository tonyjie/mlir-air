---
name: independent-evaluator
description: Phase 7 of LLM deployment — spawn a fresh subagent that treats the deployment as UNTRUSTED, audits the `make verify` implementation (anti-reward-hacking), then re-runs it as the primary numerical gate. Produces a structured `evaluation_report.md` a human can read in 2 minutes to know the full deployment state. Invoke as `/independent-evaluator <model_dir>` or auto-spawn from deploy-new-llm after Phase 6 PASS.
---

## Purpose

The deploy-new-llm chain is autonomous. Phase 6 produces a `make verify`
gate that's supposed to numerically compare NPU vs CPU reference. But
the deployment agent wrote both the gate AND the implementation —
nothing structurally prevents the gate from being mocked, hand-waved,
or measured against the wrong baseline. Phase 7 closes this by:

1. **Spawning a FRESH subagent** with no context from the deployment
2. **Auditing the `make verify` code path** to confirm it really
   computes cosine vs CPU reference (anti-reward-hacking)
3. **Re-running `make verify` independently** — the primary gate
4. Layering on adversarial + anti-fallback sanity checks
5. **Producing a structured evaluation report** humans can read in
   2 minutes to know the full deployment state

The deployment is NOT considered trustworthy until this report is
written and its overall verdict is PASS or PASS-with-warnings.

## Phase 7 PASS criteria (HARD GATES)

1. **`make verify` audited**: subagent has read the Makefile target
   AND the script it invokes AND confirmed it actually computes
   `cosine(npu_logits, cpu_reference_logits)` against
   `<model>_reference.py` (not against a cached / mocked / hand-edited
   value). Reward-hacking smell test.
2. **`make verify` PASSES under fresh subagent run**: Phase 6's gate
   re-runs cleanly — Phase 3 numerical checks + multi-token greedy
   match.
3. **`make run` reproducible (twice)**: byte-identical generated text
   across two runs (greedy decode is deterministic). Variance per-
   trial recorded.
4. **Adversarial prompts pass**: 2-3 prompts NOT in the canonical set
   verify NPU greedy == CPU greedy (catches over-tuning to canonical
   prompts).
5. **Anti-fallback heuristics pass**: kernels really fired (per-kernel
   ms within expected range; kernel cache exists; cold/warm gap
   observed).
6. **Evaluation report written** in the structured template (see Step
   8 / reference example below). Verdict PASS or PASS-with-warnings.
7. (Optional, conditional) **Cross-deployment regression PASSES** if
   shared infra changed in this deployment.

If the subagent declares PASS without showing measured numbers OR
without auditing the verify code path, the report is REJECTED — re-spawn
with stricter instructions.

## Knowledge base references

- `programming_examples/llama32_3b/docs/evaluation_report_2026-04-26.md`
  — **the canonical evaluation report format** to follow. Read this
  BEFORE writing your own to understand the structure depth users
  expect.
- `programming_examples/llama3/docs/evaluation_report.md` — second
  reference example (Llama-3.2-1B golden deployment)
- `programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
  — Phase 1 catalog (subagent compares its measurements against this)
- `programming_examples/_llm_shared/docs/kernel_registry/supported_kernels.md`
  — kernel-by-kernel ground truth

## Workflow

### Step 1: Spawn fresh subagent with independence + audit constraints

Use the `general-purpose` Agent type. Critical instructions in the
spawn prompt:

- **Independence**: do NOT read `<model>/docs/development_progress/{LESSONS,progress,phaseN_*}.md`
  BEFORE forming your own measurements. You may CITE them AFTER
  measuring (compare your numbers vs claimed).
- **Re-derivation**: every PASS/FAIL verdict must be backed by a number
  YOU measured or a code path YOU read during this audit, not a number
  copied from a deployment doc.
- **Reward-hacking smell test**: if the deployment-claimed numbers
  look surprisingly good, audit the gate implementation FIRST — does
  `make verify` really compare against the CPU reference, or did the
  agent shortcut to "if NPU output exists → PASS"?

### Step 2: Audit `make verify` — anti-reward-hacking

BEFORE running anything, READ:

1. **The Makefile**: what command does `make verify` invoke?
   ```bash
   grep -A 5 "^verify:" <model_dir>/Makefile
   ```
2. **The invoked script** (typically `<model>_inference.py --verify`):
   what does the `--verify` code path actually do?
3. **Confirm**:
   - It loads `<model>_reference.py` and runs CPU forward on the same
     input as NPU
   - It computes `cosine(npu_logits, cpu_logits)` with a real numpy
     dot product (not a hardcoded `return 1.0`)
   - It tests multi-token greedy match (Phase 6's addition) — the
     loop calls NPU `decode` AND CPU reference's equivalent for N
     tokens and asserts token IDs match
   - The PASS threshold is sane (≥ 0.95 for final logits cos, NOT
     `> 0` or `> 0.1`)
4. **If any of (3) fails**: report this as `[FAIL] verify gate is
   reward-hacked` and stop here. Tag the deployment as
   needs-remediation.

### Step 3: Run `make verify` — the PRIMARY numerical gate

```bash
cd <model_dir>
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

Expected output (per Phase 6 design):

- Per-layer cosine table (each layer ≥ 0.85, no cliff > 0.05)
- Final logits cosine (≥ 0.95)
- Top-1 strict match for decisive prompts
- Multi-token greedy match (NPU vs CPU greedy, all N tokens identical)

If any check fails: record exact failure + cite which Phase 3 / Phase 6
gate it violated. Verdict = FAIL.

### Step 4: `make run` × 2 reproducibility + perf capture

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30
```

Verify:

- Both runs complete without traceback
- Generated text is byte-identical between runs (greedy → deterministic)
- Capture TTFT (prefill ms) + TPS (tokens/sec) from each run
- If 2nd run is significantly faster than 1st, that's expected (kernel
  cache hit). If they're the same speed AND fast (<5% gap), kernels may
  have been compiled before this audit; that's fine
- If they're the same speed AND surprisingly FAST (e.g., prefill
  < 100 ms for 16+ layers), see anti-fallback (Step 6)

### Step 5: Adversarial prompts (catch over-tuning)

Run NPU + CPU greedy on 2-3 prompts NOT in the canonical set. Examples:

- `"Light travels at"` (expected ` the` or ` approximately`)
- `"DNA stands for"` (expected ` deoxy`)
- `"The Pacific Ocean is the"` (expected ` largest`)

For each: NPU first token must be in CPU's top-5, AND CPU first token
in NPU's top-5. Catches the case where the deployment passed canonical
prompts by happenstance but doesn't generalize.

### Step 6: Anti-fallback heuristics

Verify kernels actually fire on NPU (not silent CPU fallback):

- Check `<model_dir>/build/{prefill,decode}_kernel_cache/` exists and
  contains `.elf` files
- Per-kernel ms sanity: per-layer prefill ms should be > 5 ms (a CPU
  forward at the same shape is much slower; "kernel didn't actually
  run" looks like 0.01 ms = cache load only)
- For NPU FA: compare `cpu_attn=False` (NPU FA) vs `--cpu-attn` flag
  (CPU). NPU should be ≥ 1.5× faster. If parity, kernel may not have
  fired
- LM-head GEMV: should be > 10 ms (typical 14–22 ms). If < 5 ms,
  check whether it actually ran on NPU vs CPU softmax shortcut

### Step 7: (Conditional) Cross-deployment regression

Trigger only if the recent diff touched shared infra. Quick check:

```bash
git diff main..HEAD --name-only | grep -E "^programming_examples/(_llm_shared/|matrix_vector_multiplication/|llama3/(llama3_|multi_launch_builder/))"
```

If matches: re-run `make verify` on EVERY OTHER deployment under
`programming_examples/<model>/`. NPU is a singleton — run sequentially
with `flock`. Budget ~5 min per deployment. If ANY other deployment's
`make verify` regresses, FAIL Category 7 and require the shared-infra
change to be reverted or fixed.

If no shared infra changed: mark N/A.

### Step 8: Write the evaluation report

Output: `<model_dir>/docs/evaluation_report.md`. **Follow the
structure of the canonical example**:
`programming_examples/llama32_3b/docs/evaluation_report_2026-04-26.md`.

Required sections (in this order):

```markdown
# Evaluation Report: <Model> on NPU2

**Reference deployment**: `<reference>/` — what was inherited; this
report covers what's different.

## 1. Current Status

### Verified ✓ (<date> — N of M protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: independent-evaluator`) | <verdict>: <one-line summary> |
| `make run` smoke | First token `<token>` (id=...). N-trial mean prefill <X> s ± <Y> ms |
| `make verify` (NPU vs CPU F32 reference) | NPU top-1 == CPU top-1. Final logits cosine <X>. Per-layer K/V drift <X>→<Y> over N layers. |
| HuggingFace F32 cross-check on CPU reference | top-1 `<token>`, logits correlation > 0.9999 vs HF |
| Code review | <one-line: clean / silent-fallback / etc.> |

### Performance (<N>-trial mean)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (<N> layers, ...) | X ms/layer | **<Y> s ± <Z> ms** |
| Decode steady-state | X ms/layer | **<Y> ms/token** (<Z> tok/s) |

### Manual Verify Commands

```bash
cd <model_dir>
flock ... make verify
flock ... make run N_TOKENS=30 PROMPT="..."
# Expected first token: ...
# Expected prefill:    ...
# Expected decode:     ...
```

## 2. Architectural Differences vs Reference Deployment

| Field | <Ref Model> | <This Model> | Why it matters |
|---|---:|---:|---|
| n_layers | ... | ... | ... |
| ... | ... | ... | ... |

**The single delta that matters**: <one-line summary>.

## 3. Implementation: Reused vs New

| What | Reused from | New (model-specific) |
|---|---|---|
| Per-layer prefill orchestration | <ref> | <if applicable> |
| ... | ... | ... |

## 4. End-to-End Inference Workflow

### Setup (one-time)
[code-block trace of compile + weight load + BO preload]

### Prefill — runs N times, then once at end
[per-layer XRT call breakdown with ascii boxes for each ELF]

### Decode — per token
[per-layer XRT call breakdown]

### What's on NPU vs CPU
[bulleted lists]

## Notes

- Why per-layer K/V cosine drift looks reasonable here
- Anything redundant vs reference deployment
- Recent fixes worth flagging

## File Map

| File | Role | Lines |
|---|---|---:|
| `<model>_inference.py` | ... | ... |
| ... | ... | ... |
```

The structure is rigid because human reviewers expect to find specific
information in specific places. Don't reorder sections.

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| `make verify` script doesn't actually call CPU reference (Step 2 audit fails) | Reward-hacked gate | Report `[FAIL] verify gate is reward-hacked`; deployment needs remediation; do NOT mark PASS |
| `make verify` runs but returns PASS suspiciously fast (<10 s for full N-layer model) | Verify only spot-checks logits, doesn't run multi-token greedy | Inspect script; if multi-token loop missing → `[FAIL] verify is incomplete` |
| `make run` non-deterministic across two runs (greedy) | Sampling enabled by mistake OR uninitialized BO state | `[FAIL]`; bisect to find which call introduces non-determinism |
| Adversarial prompt: NPU top-1 NOT in CPU top-5 | Real correctness issue OR over-tuning to canonical set | Report measurement; verdict depends on severity (one fail = WARN, multiple = FAIL) |
| Per-kernel ms surprisingly low (LM head < 5 ms) | Kernel didn't actually run (silent CPU fallback) | Check kernel cache files exist; flag if missing |
| Subagent reads LESSONS/progress before measuring | Skill prompt wasn't strict enough | Reject the report; re-spawn with stricter instructions |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

This is the terminal verification phase. On Phase 7 PASS or
PASS-with-warnings:

- `<model_dir>/docs/evaluation_report.md` is the durable artifact
- Append to `<model>/TODO.md`: "Independently evaluated YYYY-MM-DD: <verdict>"
- Reference the report from `<model>/docs/development_progress/progress.md`

If FAIL: deployment cannot be tagged. Issues must be remediated and
Phase 7 re-run.
