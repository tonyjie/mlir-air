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

### Step 4: Scaffold `<model>/` directory — minimal, sys.path-imports

**Do NOT `cp -r llama3 <model>`.** That copies ~500 KB of unused `llama3_*.py`,
`multi_launch_builder/`, and `test/` files that the per-model scripts never
import (they resolve via sys.path to `../llama3/`). Stale duplicates cause
confusion as the shared code evolves and bloat git blame.

The minimal Tier-A scaffold is **6 files**:

```
<dirname>/
├── .gitignore                       # copy from llama3/.gitignore + add *.o, *kernel_cache/
├── Makefile                         # template-render with model name
├── README.md                        # placeholder; final version written by finalize-deployment
├── CLAUDE.md                        # model-specific guide (template below)
├── TODO.md                          # phase status (template in Step 5)
└── docs/development_progress/
    ├── progress.md                  # header-only
    ├── LESSONS.md                   # header-only
    └── debug_log.md                 # header-only
```

**At runtime**, every per-model script needs this sys.path block to resolve
imports from the llama3 reference and `_llm_shared/`:

```python
from pathlib import Path
import sys
_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
for p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Now these resolve to ../llama3/ and ../_llm_shared/:
from llama3_prefill import KernelCache, run_transformer_block, ...
from llama3_decode import run_decode_block, compile_decode_kernels
import llama3_inference  # for _preload_decode_weights, etc.
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
```

**Per-phase skills produce these SmolLM2-specific files in this dir**:
- Phase 0 (`bootstrap-model-config`): `<model>_weights.py`, `<model>_reference.py`
- Phases 2-5: `<model>_phaseN_test.py` (one per validation phase)
- Post-Phase 5 (or in `finalize-deployment`): `<model>_inference.py` —
  the end-to-end NPU runner (NPU prefill with K/V extraction → NPU LM Head →
  NPU decode loop). Modeled on `llama3_inference.run_npu_prefill` + `generate`.

**Makefile template** (mirror llama3's UX, llama3-style targets):
- `make compile` (all kernels) / `compile-prefill` / `compile-decode`
- `make run` → `<model>_inference.py` (end-to-end NPU)
- `make profile` → prefill perf
- `make verify` → decode with NPU/CPU top-1 check
- `make run-{block,full,prefill,reference,decode-only}` → individual phase scripts
- Env vars: `PROMPT`, `N_TOKENS`, `SEQ_LEN`, `MAX_SEQ`, `MODEL` plumbed through
- `make clean` → remove `*kernel_cache/`, `air_project/`, `build_*/`, `*.o`, etc.
- See `programming_examples/smollm2_1_7b/Makefile` for the canonical template.

**Why minimal scaffold + sys.path imports**:
- Lesson from smollm2_1_7b deployment (2026-04-17): a `cp -r` scaffold copied
  47 files, only 22 of which were actually used. The 25 unused files
  (`llama3_*.py`, `multi_launch_builder/`, `test/`) caused these issues:
  1. `python -c "from llama3_prefill import ..."` resolved to the LOCAL stale
     copy (because `_THIS_DIR` is first in sys.path), not the shared `../llama3/`
     code — silently using outdated logic.
  2. Bug fixes in `../llama3/` wouldn't propagate to the per-model dir.
  3. ~500 KB of duplicated code per deployment.
- The minimal-scaffold pattern with sys.path imports keeps per-model dirs
  small (~22 files), guarantees freshness from `../llama3/`, and makes
  divergences in per-model code explicit (you only see what's actually
  SmolLM2-specific).
- For models that NEED to fork llama3_prefill (true arch divergence), copy
  ONLY the file being forked, rename it `<model>_prefill.py`, and import the
  rest from `../llama3/`. Don't bulk-copy.

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
