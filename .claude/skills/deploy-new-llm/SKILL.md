---
name: deploy-new-llm
description: Entry point for deploying a new decoder-only LLM on AMD NPU2. Invoked by the user as `/deploy-new-llm <hf_model_id> [--name <dirname>] [--target npu2|npu1] [--dtype bf16|fp16]`. Bootstraps the per-model workspace, validates architecture is in scope, and walks through the 7-phase deployment workflow with human-approved gates between phases.
---

## Purpose
Single user-facing entry point that scaffolds a new model deployment and orchestrates the per-phase skills with verification gates between them.

## Knowledge base references
- `docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md` — the design spec this entry skill implements
- `programming_examples/llama3/` — the reference deployment

## Workflow

### Step 1: Parse arguments
- Required: HF model ID (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Optional: `--name <dirname>` (default: derived from model ID, lowercased, slashes → underscores)
- Optional: `--target npu2|npu1` (default: `npu2`)
- Optional: `--dtype bf16|fp16` (default: `bf16`)

### Step 2: Architecture compatibility check
Fetch HF `config.json`. Reject if any of:
- Architecture is MoE (e.g., `MixtralForCausalLM`, gpt-oss class)
- Has sliding-window attention (`sliding_window` set in config)
- Uses MLA (Multi-head Latent Attention)
- Uses encoder-decoder structure
- Has QKV bias without bias-supporting kernels available

Print clear rejection message. Do NOT proceed.

### Step 3: Check for `_llm_shared/`
```bash
test -d /home/jiajli/apps/mlir-air/programming_examples/_llm_shared && echo "_llm_shared exists" || echo "_llm_shared missing"
```

If missing, the one-time lift refactor has not been done. **Halt and instruct the human:** "Run the lift refactor first (see Plan Task 2). This is a one-time setup that lifts `kernel_builder/` from `llama3/` to `_llm_shared/`."

### Step 4: Scaffold `<model>/` directory
```bash
cp -r /home/jiajli/apps/mlir-air/programming_examples/llama3 /home/jiajli/apps/mlir-air/programming_examples/<dirname>
```

Then within `<dirname>/`:
- Rename `llama3_*.py` files → `<model>_*.py` (preserve content for now; phase 0 will adapt)
- Update Makefile model-name references
- Clear `prefill_kernel_cache/` and `build_peano/` (they'll be regenerated)

### Step 5: Initialize `<model>/TODO.md`
Create with this template (filled in with config from Step 2):

```markdown
# Deployment: <model_name>

## Phase status
- [ ] 0: Bootstrap
- [ ] 1: Per-kernel shapes
- [ ] 2: Single block
- [ ] 3: Full model
- [ ] 4: Prefill perf
- [ ] 5: Decode perf
- [ ] 6: Finalize

## Active blockers
(none yet)

## Resolved config (pulled from HF)
n_layers: <N>, emb_dim: <D>, n_heads: <H>, n_kv_heads: <K>,
head_dim: <hd>, hidden_dim: <F>, vocab_size: <V>, rope_theta: <R>
```

### Step 6: Initialize per-model docs
Create `<model>/docs/development_progress/`:
- `progress.md` (header only; phases append as they pass)
- `LESSONS.md` (header only; appended on novel failures)
- `debug_log.md` (header only; appended on debug-recipe firings)

### Step 7: Walk human through phases
Report current state to human:
> "Workspace scaffolded at `programming_examples/<dirname>/`. Resolved config: <summary>. Ready to start Phase 0 (Bootstrap). Invoke `bootstrap-model-config` to begin, or say 'go' for me to invoke it now."

For each phase:
1. Invoke the per-phase skill
2. Wait for the skill to complete or escalate
3. Report PASS/FAIL/BLOCKED to human
4. On PASS: ask permission to advance to next phase
5. On BLOCKED: surface the blocker, human resolves
6. Advance to next phase

### Step 8: On final PASS, hand off
Once Phase 6 (`finalize-deployment`) is PASS, report:
> "Deployment complete. See `programming_examples/<dirname>/docs/development_progress/progress.md` for the summary."

## Verification

The entry skill itself doesn't have a "gate" — its success is measured by the per-phase gates. The entry skill is "successful" when all 7 phases reach PASS.

## Failure modes
- Architecture rejected at Step 2 → no scaffolding done; deployment cannot proceed
- `_llm_shared/` missing at Step 3 → human must run the lift first
- Per-phase gate failures → handled by the respective per-phase skill, surfaced to human via TODO.md

## Update protocol
The entry skill primarily reads TODO.md and dispatches; it doesn't write to progress files itself (per-phase skills do that).
