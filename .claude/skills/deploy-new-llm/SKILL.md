---
name: deploy-new-llm
description: Entry point for deploying a new decoder-only LLM on AMD NPU2. Invoked by the user as `/deploy-new-llm <hf_model_id> [--name <dirname>] [--target npu2|npu1] [--dtype bf16|fp16]`. Bootstraps the per-model workspace, validates architecture is in scope, and dispatches the 7 per-phase skills with the gate of each phase enforced by that phase's skill.
---

## Purpose

Single user-facing entry point that scaffolds a new model deployment
and orchestrates the per-phase skills. The orchestrator does NOT do
correctness work itself — every gate is enforced by the corresponding
per-phase skill. This skill's job is workflow coordination + workspace
bootstrap.

## Orchestrator success criteria

This skill is "successful" when:

1. **Workspace scaffolded** correctly (Steps 1-6 below complete)
2. **Phases 0-6 dispatched in order**, each phase's HARD gate (defined
   inside the per-phase SKILL.md) passes
3. **Phase 7 (`independent-evaluator`) verdict** = PASS or
   PASS-with-warnings
4. **Hand-off report written** to the human (Step 9)

If any phase's gate fails irrecoverably, the deployment is marked
`needs-human-review` in TODO.md and the orchestrator stops — the human
triages.

## Knowledge base references

- `programming_examples/llama3/` — the reference Tier-A deployment
  (everything in scope today inherits from this)
- `programming_examples/_llm_shared/docs/kernel_registry/supported_kernels.md`
  — kernel registry; Phase 1 will populate this model's catalog under
  the same dir
- `programming_examples/_llm_shared/docs/kernel_registry/llama3.2_1b.md`
  — per-model shape catalog format (template Phase 1 mirrors)
- `docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md`
  — original design spec for the skill chain

## Workflow

### Step 0: Preconditions check

This skill assumes the user already has:

- **mlir-air built and the environment sourced** (project's
  `SessionStart` hook in `.claude/settings.local.json` handles this
  automatically; for manual shells see `.claude/CLAUDE.md` "Environment
  Setup"). Smoke test: `cd programming_examples/llama3 && make help`
  prints the target list.
- **NPU2 hardware accessible via XRT** (no other process holding it).
- **HuggingFace login + model access**. For gated models like
  `meta-llama/Llama-3.2-3B`, run `huggingface-cli login` and accept the
  model card on huggingface.co before invoking this skill. ~6 GB disk
  per BF16 3B model in `~/.cache/huggingface/hub/`.
- **System DRAM ≥ 16 GB** for 1-3 B models; deeper deployments
  approach the limit.

If any are missing, halt and ask the user to address them. Do NOT try
to install MLIR-AIR / set up XRT / log into HF on the user's behalf.

### Step 1: Parse arguments

- Required: HF model ID (e.g., `meta-llama/Llama-3.2-3B`)
- Optional: `--name <dirname>` (default: derived from model ID,
  lowercased, slashes → underscores)
- Optional: `--target npu2|npu1` (default: `npu2`)
- Optional: `--dtype bf16|fp16` (default: `bf16`)

### Step 2: Architecture compatibility check

Fetch HF `config.json`. Reject if any of:

- Architecture is MoE (e.g., `MixtralForCausalLM`, gpt-oss class)
- Has sliding-window attention (`sliding_window` set in config AND
  `use_sliding_window=true`)
- Uses MLA (Multi-head Latent Attention)
- Uses encoder-decoder structure

**QKV bias is supported** (Qwen2-family models with `qkv_bias=true`):
the bias is added on the HOST after the bias-free kernels return,
exploiting RoPE's linearity (`RoPE(q + bq) = RoPE(q) + RoPE(bq)`).
Reference: `programming_examples/qwen25_1_5b/qwen25_bias.py`.
Per-deployment effort: ~1-2 hours; surface in TODO.md as a Phase 2
prerequisite.

If rejected, print clear message and do NOT proceed.

### Step 3: Check for `_llm_shared/`

```bash
test -d programming_examples/_llm_shared && echo OK || echo MISSING
```

If missing, the one-time lift refactor has not been done. Halt and
instruct the human.

### Step 4: Scaffold `<model>/` directory — minimal, sys.path-imports

**Do NOT `cp -r llama3 <model>`.** That copies ~500 KB of unused
`llama3_*.py`, `multi_launch_builder/`, and `test/` files that the
per-model scripts never import (they resolve via sys.path to
`../llama3/`). Stale duplicates cause:

- `from llama3_prefill import ...` resolving to the LOCAL stale copy,
  not the shared `../llama3/` — silently using outdated logic
- Bug fixes in `../llama3/` not propagating to the per-model dir
- ~500 KB duplicated code per deployment

Lesson learned from smollm2_1_7b deployment: `cp -r` scaffolded 47
files, only 22 used; the 25 stale files caused all of the above.

The minimal Tier-A scaffold is **6 files**:

```
<dirname>/
├── .gitignore                       # copy from llama3/.gitignore + add *.o, *kernel_cache/
├── Makefile                         # template-render with model name (3 targets: run / verify / profile + compile / clean)
├── README.md                        # placeholder; final version written by finalize-and-learn
├── CLAUDE.md                        # model-specific guide
├── TODO.md                          # phase status (template in Step 5)
└── docs/development_progress/
    ├── progress.md                  # header-only; phases append as they pass
    ├── LESSONS.md                   # header-only; appended on novel failures
    └── debug_log.md                 # header-only; appended on debug-recipe firings
```

**Per-model scripts use this sys.path block** to resolve imports from
the llama3 reference and `_llm_shared/`:

```python
from pathlib import Path
import sys
_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
for p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Resolves to ../llama3/ and ../_llm_shared/:
from llama3_prefill import KernelCache, run_transformer_block, ...
from llama3_decode import run_decode_block, compile_decode_kernels
import llama3_inference  # for _preload_decode_weights, etc.
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
```

For models that NEED to fork llama3_prefill (true arch divergence),
copy ONLY the file being forked, rename it `<model>_prefill.py`, and
import the rest from `../llama3/`. Don't bulk-copy.

**Per-phase skills produce these model-specific files**:

- Phase 0 (`build-cpu-oracle`): `<model>_weights.py`, `<model>_reference.py`
- Phases 1-3 (validation): `<model>_phaseN_test.py` per phase
- Phase 6 (`finalize-and-learn`): `<model>_inference.py` — clean
  end-to-end NPU runner (setup → prefill → decode_loop)

**Makefile template** (mirror `programming_examples/smollm2_1_7b/Makefile`):

- `make compile` — compile all kernels
- `make run` — `<model>_inference.py --n-tokens 100`
- `make verify` — `<model>_inference.py --verify`
- `make profile` — `<model>_inference.py --profile`
- `make run-{block,full,prefill,decode-only}` — individual phase scripts
- Env vars: `PROMPT`, `N_TOKENS`, `SEQ_LEN`, `MAX_SEQ` plumbed through
- `make clean` — remove `*kernel_cache/`, `air_project/`, `build_*/`, `*.o`

### Step 5: Initialize `<model>/TODO.md`

Template (filled with Step 2 config):

```markdown
# Deployment: <model_name>

## Phase status
- [ ] 0: Build CPU Oracle
- [ ] 1: Kernel Validation
- [ ] 2: Single-Block Validation
- [ ] 3: Full-Model Validation
- [ ] 4: Prefill Optimization
- [ ] 5: Decode Optimization
- [ ] 6: Finalize & Learn
- [ ] 7: Independent Evaluation

## Active blockers
(none yet)

## Resolved config (pulled from HF)
n_layers: <N>, emb_dim: <D>, n_heads: <H>, n_kv_heads: <K>,
head_dim: <hd>, hidden_dim: <F>, vocab_size: <V>, rope_theta: <R>
```

### Step 6: Initialize per-model docs

Create `<model>/docs/development_progress/`:

- `progress.md` (header only)
- `LESSONS.md` (header only)
- `debug_log.md` (header only)

### Step 7: Dispatch the 7 phases

**Phase → skill mapping** (each gate enforced by the per-phase skill):

| Phase | Skill | Gate (in 1 line — see the skill itself for full criteria) |
|---|---|---|
| 0 | `build-cpu-oracle` | `<model>_reference.py` matches HF transformers (per-layer cos ≥ 0.99 + final logits cos ≥ 0.999 + top-1 strict) |
| 1 | `kernel-validation` | Every leaf kernel × shape: standalone NPU test PASSES cosine vs CPU; recorded in `kernel_registry/<model>.md` |
| 2 | `single-block-validation` | Single transformer block on NPU: cosine vs CPU reference ≥ 0.99 (whole-tensor) + per-position min ≥ head_dim-scaled threshold |
| 3 | `full-model-validation` | Full N layers: per-layer cos ≥ 0.85 + no cliff + final logits cos ≥ 0.95 + top-1 strict (decisive) / top-5 overlap (competitive) |
| 4 | `prefill-optimization` | Apply optimization patterns; correctness preserved (Phase 3 re-run) AND prefill kernel time strictly < Phase 3 baseline |
| 5 | `decode-optimization` | Same shape: correctness preserved AND decode time/token strictly < Phase 4 baseline |
| 6 | `finalize-and-learn` | Clean `<model>_inference.py` + Makefile (run/verify/profile); `make verify` PASSES strict numerical check vs CPU + multi-token greedy match |
| 7 | `independent-evaluator` | Fresh subagent: audit `make verify` (anti-reward-hacking) + re-run as primary gate; produce structured `evaluation_report.md` |

Report current state to the human:

> "Workspace scaffolded at `programming_examples/<dirname>/`. Resolved
> config: <summary>. Ready to start Phase 0 (Build CPU Oracle). Invoke
> `build-cpu-oracle` to begin, or say 'go' for me to invoke it now."

For each phase (0 → 6):

1. Invoke the per-phase skill from the table
2. Wait for the skill to complete or escalate
3. Report PASS/FAIL/BLOCKED to the human
4. On PASS: ask permission to advance to next phase
5. On BLOCKED: surface the blocker, human resolves
6. Advance to the next phase

### Step 8: Phase 7 — Independent evaluation

After Phase 6 PASSES but BEFORE the final hand-off, spawn the
`independent-evaluator` skill as Phase 7. It re-derives every
correctness claim with a fresh subagent and produces
`<model>/docs/evaluation_report.md`.

Why: Phases 0-6 are autonomous and self-reporting. The deployment
agent has no incentive to cheat, but also no incentive to catch its
own silent regressions (preload errors, fallback gates, etc.). Phase 7
is the independence check.

> "Spawning Phase 7 — independent evaluation. The evaluator subagent
> will audit `make verify` (anti-reward-hacking), re-run it as the
> primary gate, and write a structured report. Expected runtime:
> 15-30 min."

Then call the `independent-evaluator` skill with `<model_dir>` as input.

If the evaluator reports:

- **PASS** → proceed to Step 9
- **PASS-with-warnings** → proceed to Step 9; warnings go in TODO.md
  as "follow-up"
- **FAIL** → mark deployment `needs-human-review` in TODO.md and STOP.
  Do NOT hand off. Surface specific failures.

If the deployment touched shared infra (any of `_llm_shared/`,
`matvec.py`, `llama3/multi_launch_builder/`, `llama3/llama3_*.py`),
the evaluator's Step 7 (Conditional Cross-deployment regression) will
also re-verify every OTHER deployment's `make verify` to catch
back-compat breaks. Budget ~5 min per deployment.

### Step 9: On all-PASS, hand off to the human

Once Phase 6 PASS AND Phase 7 PASS (or PASS-with-warnings), report:

> "Deployment complete. See:
> - `programming_examples/<dirname>/docs/development_progress/progress.md` — phase summary
> - `programming_examples/<dirname>/docs/evaluation_report.md` — independent audit
> - `programming_examples/_llm_shared/docs/kernel_registry/<dirname>.md` — kernel × shape catalog"

Optional: tag the deployment if the project workflow uses git tags
(`git tag -a deployment-<dirname>-v1 -m "..."`). Most deployments
don't tag — git log + commit messages are the durable record.

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| Architecture rejected at Step 2 | MoE / sliding-window / MLA / encoder-decoder model | Halt; tell the user this model is out of scope |
| `_llm_shared/` missing at Step 3 | One-time lift refactor not done | Halt; instruct human to run the lift |
| Per-phase gate fails | Per-phase skill should escalate via TODO.md "Active blockers" | Don't try to fix here; the per-phase skill's failure-mode table is the right place |
| Phase 7 = FAIL | Evaluator surfaced a real correctness or reward-hacking issue | Mark `needs-human-review` in TODO.md; STOP; do NOT hand off |
| Cross-deployment regression at Phase 7 | This deployment's shared-infra change broke another deployment | Revert or fix the shared-infra change before tagging |
| User skips a phase to "save time" | Skipped phases mean later phases verify against unverified upstream | Refuse to advance past the skipped gate; explain the dependency chain (Phase 1 → 2 → 3 → 4/5 → 6 each need the previous PASS) |

For any failure not in the table, escalate to the human (this skill is
orchestration; debugging belongs in the per-phase skills' failure-mode
tables or the cross-cutting `debug-*` skills).

## Update protocol

This skill primarily reads `TODO.md` and dispatches; it doesn't write
to progress files itself (per-phase skills do that). On all-PASS:

- `<model>/TODO.md` reflects all 7 phases checked
- `<model>/docs/development_progress/progress.md` has each phase's
  summary entry (written by per-phase skills)
- `<model>/docs/evaluation_report.md` exists (written by Phase 7)
- (optional) git tag created
