---
name: deploy-scaffold-reference
description: Knowledge base for deploying a new decoder-only LLM on NPU2 — architecture scope rules, workspace scaffold layout, and per-model TODO/docs templates. This is a REFERENCE consulted by the deploy agent team (deploy-researcher / deploy-planner / deploy-runner), NOT a user entry point. The entry point is the `/deploy-new-llm` command (the multi-agent orchestrator). Use when scaffolding a new model directory or checking whether an architecture is in scope.
---

## Purpose

Reference material for standing up a new model deployment: which architectures are
in scope, how the `<model>/` workspace is laid out, and the templates for its
TODO.md / per-phase docs. **This is not the entry point and does not orchestrate
anything** — the `/deploy-new-llm` command (a multi-agent orchestrator) owns
workflow coordination, and the `deploy-*` agents own execution and verification
against `.claude/checklists/deploy-parity-checklist.md`. Those agents consult this
file for the scaffold/scope details below.

Historical note: this used to be the single-agent orchestrator skill. Its dispatch
logic was replaced by the agent team — the old single-agent chain shipped
correct-but-slow models (kernels silently on CPU, ELFs left unmerged) because one
agent optimizing for the correctness gate had no independent check on execution
location or merge completeness. What remains here is the durable, reusable part:
scope rules + scaffold + templates.

## What consumes this reference

- **deploy-researcher** — the architecture scope rules (Step 2 below) + how to
  pick the sibling template.
- **deploy-planner / deploy-runner** — the scaffold layout (Step 4) and the
  TODO/docs templates (Steps 5-6).
- The **`/deploy-new-llm`** orchestrator command owns dispatch; the old "Step 7-9
  dispatch/hand-off" flow that used to live here is gone (the command + agents do
  it now).

## Knowledge base references

- `programming_examples/llms/llama32_1b/` — the reference Tier-A deployment
  (everything in scope today inherits from this)
- `programming_examples/llms/verify/` — the **shared** verify subsystem
  (HF bf16 reference; `make verify` token-set check is the PASS/FAIL gate,
  `make diagnosis` per-layer cosine is the informational lens); every model
  hooks in via its own `verify_adapter.py`, not by copying this.
- `programming_examples/llms/shared/infra/` — the **shared** kernel
  builder (KernelCache, external kernels, stitching, the `ffn_swiglu/`
  harness); per-model scripts import from it.
- `programming_examples/kernel_registry/` — the model-agnostic kernel
  registry. Human half: `supported_kernels.md` (index) + `details/<Kernel>_bf16.md`
  per kernel (GEMM is split by output dtype: `GEMM_bf16_in_bf16_out.md` +
  `GEMM_bf16_in_fp32_out.md`). Machine half: `details/*.json` +
  `registry_lookup.py` (`gemm_config(...)` returns the best measured tile
  config; raises for unmeasured shapes). Phase 1 appends this model's
  verified (kernel, shape) rows (Used by = `<model>`); Phase 4 builders read
  tile configs back via the lookup instead of hardcoding them.

## Workflow

### Step 0: Preconditions check

This skill assumes the user already has:

- **mlir-air built and the environment sourced** (see the repo build
  docs; for manual shells, source the mlir-air env before invoking).
  Smoke test: `cd programming_examples/llms/llama32_1b && make help` prints
  the target list.
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

**QKV bias is supported** (e.g. Qwen2-family with `qkv_bias=true`) — added
on the host around the bias-free kernels; the technique lives in
`phase-2-single-block-validation` Step 2. Surface it in TODO.md as a Phase 2
prerequisite.

If rejected, print clear message and do NOT proceed.

### Step 3: Check for the shared infra + reference exemplar

```bash
test -d programming_examples/llms/shared/infra && \
test -d programming_examples/llms/verify && \
test -d programming_examples/llms/llama32_1b && echo OK || echo MISSING
```

The first two are **required**: every deployment composes kernels via the
shared `shared/infra` toolkit and gates on the shared `verify/`
subsystem. The third, `llama32_1b`, is the **reference exemplar** — read
to mirror its assembly, and imported directly on a bit-for-bit match. If
any is missing, halt and instruct the human.

### Step 4: Scaffold `<model>/` directory — kernel-first, minimal

The model lives at `programming_examples/llms/<dirname>/`, a sibling of
`programming_examples/llms/llama32_1b/` (the reference exemplar) and the shared `programming_examples/llms/verify/`
+ `programming_examples/llms/shared/infra/` (the toolkit every deployment builds on).

**Default mindset: build this model up from registry kernels** using the
shared `shared/infra` toolkit (KernelCache, stitching,
external_kernels). The per-phase skills write the model's own
`<model>_prefill.py` / `<model>_decode.py` / `shared/builders/` by
composing the Phase-1-verified leaf kernels, reading `llama32_1b`'s
assembly as the worked exemplar. This generalizes to any in-scope
architecture — it does not assume the model resembles llama.

**Do NOT `cp -r llama32_1b <model>`.** Two reasons depending on path
(the kernel-first-vs-inheritance decision + the bit-for-bit match rule are
owned by `phase-2-single-block-validation` Step 1 — Phase 2 makes the call):
- Kernel-first (default): you're writing model-specific assembly, not
  copying the reference's — bulk-copying just duplicates stale code.
- Inheritance shortcut (bit-for-bit llama variant only): the reference's
  `llama32_1b_*.py` resolve via sys.path to `../llama32_1b/`, so there's
  nothing to copy; a local copy would silently use outdated logic and miss
  upstream bug fixes.

The minimal Tier-A scaffold is:

```
programming_examples/llms/<dirname>/
├── .gitignore                       # copy from llama32_1b/.gitignore + add *.o, *kernel_cache/
├── Makefile                         # template-render with model name (run / verify / verify-full / diagnosis / profile + compile / clean)
├── README.md                        # placeholder; final version written by phase-6-finalize-and-learn
├── ARCHITECTURE.md                  # model-specific guide (NOT CLAUDE.md — top-level .gitignore excludes it, so it would not ship)
├── TODO.md                          # phase status (template in Step 5)
├── verify_adapter.py                # written by phase-6-finalize-and-learn; hooks this model into the shared programming_examples/llms/verify/
└── docs/development_progress/
    ├── progress.md                  # header-only; phases append as they pass
    ├── LESSONS.md                   # header-only; appended on novel failures
    └── debug_log.md                 # header-only; appended on debug-recipe firings
```

**Per-model scripts use this sys.path block** to resolve the shared
`programming_examples/llms/` packages (always) and the llama32_1b reference (as exemplar, or
to import directly on a bit-for-bit match):

```python
from pathlib import Path
import sys
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent              # programming_examples/llms/
for p in (_LLMS_DIR, _LLMS_DIR / "llama32_1b", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ALWAYS — the shared kernel toolkit you compose this model FROM:
from shared/infra.external_kernels import compile_all_external_kernels
from shared/infra.cache import KernelCache

# DEFAULT (kernel-first) — write <model>_prefill.py / shared/builders/
# that assemble the registry leaf kernels for THIS model's sequence,
# mirroring llama32_1b's builders as the worked example.

# SHORTCUT (bit-for-bit llama variant ONLY) — skip writing builders and
# reuse the reference's assembly directly:
#   from llama32_1b_prefill import run_transformer_block, ...
#   from llama32_1b_decode import run_decode_block, compile_decode_kernels
```

For models that fork one reference module (partial arch divergence), copy
ONLY the file being forked, rename it `<model>_prefill.py`, compose the
divergent part kernel-first, and import the unchanged rest from
`../llama32_1b/`. Don't bulk-copy.

**Per-phase skills produce these model-specific files**:

- Phase 0 (`phase-0-build-cpu-reference`): `<model>_weights.py`, `<model>_cpu_helpers.py`
- Phases 1-3 (validation): `<model>_phaseN_test.py` per phase
- Phase 6 (`phase-6-finalize-and-learn`): `<model>_inference.py` (clean
  end-to-end NPU runner: setup → prefill → decode_loop) + `<model>/verify_adapter.py`
  (hooks this model into the shared `programming_examples/llms/verify/`; mirrors
  `llama32_1b/verify_adapter.py`)

**Makefile template** (mirror `programming_examples/llms/llama32_1b/Makefile`):

- `make compile` — compile all kernels
- `make run` — `<model>_inference.py --n-tokens 100`
- `make verify` / `make verify-full` — `../verify/verify_runner.py --runner=<model>.verify_adapter` token-set gate
- `make diagnosis` — `../verify/verify_runner.py --runner=<model>.verify_adapter` per-layer cosine
- `make profile` — `<model>_inference.py --profile`
- Env vars: `PROMPT`, `N_TOKENS`, `MODEL` (base/instruct) plumbed through
- `make clean` — remove `*kernel_cache/`, `air_project/`, `build_*/`, `*.o`, `verify/reports/`

### Step 5: Initialize `<model>/TODO.md`

Template (filled with Step 2 config):

```markdown
# Deployment: <model_name>

## Phase status
- [ ] 0: Build CPU Reference
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
- `phase_timing.md` (REQUIRED — per-phase wall-clock log; schema below)

**`phase_timing.md` schema** (per-phase effort breakdown — useful for
understanding which architectural axes are genuinely hard). Capture the
deployment-session start timestamp (`date +"%s"`) at scaffold time.
Update at every phase boundary:

```markdown
# <Model> deployment — per-phase wall-clock log

## Baselines

- Deployment session start: <YYYY-MM-DD HH:MM:SS TZ> (epoch=<N>)
- Scaffold complete:        <YYYY-MM-DD HH:MM:SS TZ> (epoch=<N>)

## Phase log

### Phase N — <Name>  (PENDING / PASS / PASS-with-warnings / BLOCKED, YYYY-MM-DD)

- start_ts:           <epoch s>  (HH:MM:SS TZ)
- end_ts:             <epoch s>  (HH:MM:SS TZ)
- wall_min:           **<N>**
- npu_compile_min:    <N>   (sum of NPU kernel compile times in this phase)
- npu_runtime_s:      <N>   (sum of XRTRunner / inference NPU run time)
- dev_min:            **<N>**   ≈ wall - compile - runtime (agent thinking/code/debug)
- notable_events:     <bullets — record honestly even if "stuck on debug for K min">

## Summary table  (filled at deployment end)

| Phase | wall_min | npu_compile_min | npu_runtime_s | dev_min | notes |
|---|---:|---:|---:|---:|---|
| Scaffold + Step 0-3 | | | | | |
| 0: CPU Oracle | | | | | |
| ... | | | | | |
| **Total** | | | | | |
```

**Why this matters**: `dev_min` (vs `npu_compile_min` / `npu_runtime_s`)
is the real "agentic deployment cost". Even debug-stuck phases should be
honestly recorded — high `dev_min` on a phase reveals which
architectural axes are genuinely hard, which informs future deployments.


---

## After scaffold: the agent team takes over

Once the workspace is scaffolded and the architecture is confirmed in scope, the
`/deploy-new-llm` orchestrator drives the phases via the `deploy-*` agents against
`.claude/checklists/deploy-parity-checklist.md`. Dispatch, per-phase gating,
independent verification (Gates H/I/T), and the human hand-off all live there —
not in this reference.
