# LLM Mapping Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a skill-based playbook in `.claude/skills/` for deploying decoder-only LLMs on AMD NPU2, leveraging the LLAMA-3.2-1B infrastructure in `programming_examples/llama3/`. End state: 11 skills authored + foundation refactor complete + smoke-tested by deploying Llama-3.2-1B-Instruct end-to-end.

**Architecture:** One-time lift of `kernel_builder/` from `programming_examples/llama3/` to `programming_examples/_llm_shared/` (single source of truth). Three categories of skills authored: 7 per-phase skills (bootstrap → finalize), 3 cross-cutting recipe skills (debug/merge), and 1 entry skill (`/deploy-new-llm`). Skills are short procedural markdown that cite the rich `programming_examples/llama3/docs/development_progress/` knowledge base rather than duplicating content. Validation by smoke-test on Llama-3.2-1B-Instruct (trivially identical arch, proves end-to-end skill flow runs). Extended pilots (TinyLlama-1.1B, SmolLM2-1.7B) are explicit follow-on work outside this plan.

**Tech Stack:** Markdown (skills), Python 3 (existing kernel_builder), MLIR-AIR (existing pipeline), XRT (existing runtime), AMD NPU2 (Strix). No new dependencies introduced.

**Spec:** `docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md`

**Out of scope for this plan:**
- Real pilot on TinyLlama-1.1B (deferred — discovery-style work, will iterate skills as friction surfaces)
- Broader test on SmolLM2-1.7B (deferred — depends on TinyLlama findings)
- Any MoE / sliding-window / MLA support (out of spec scope)

---

## File Structure

**Created files (skills, project-versioned):**

```
.claude/skills/
├── deploy-new-llm/SKILL.md
├── bootstrap-model-config/SKILL.md
├── validate-per-kernel-shapes/SKILL.md
├── integrate-single-block/SKILL.md
├── validate-full-model-correctness/SKILL.md
├── optimize-prefill-perf/SKILL.md
├── optimize-decode-perf/SKILL.md
├── finalize-deployment/SKILL.md
├── debug-bo-corruption/SKILL.md
├── debug-multi-launch-merge/SKILL.md
└── merge-multi-launch-kernels/SKILL.md
```

**Created files (shared infra):**

```
programming_examples/_llm_shared/
├── kernel_builder/         # moved from llama3/kernel_builder/
│   ├── __init__.py
│   ├── cache.py
│   ├── stitching.py
│   ├── gemm_builder.py
│   ├── external_kernels.py
│   ├── rope_halfsplit.cc
│   └── ffn_swiglu/
└── README.md               # how to use _llm_shared
```

**Modified files:**

```
programming_examples/llama3/llama3_inference.py    # imports rewritten
programming_examples/llama3/llama3_decode.py       # imports rewritten
programming_examples/llama3/llama3_prefill.py      # imports rewritten (if any)
programming_examples/llama3/multi_launch_builder/lm_head_gemv_multi.py
programming_examples/llama3/multi_launch_builder/lm_head_multi.py
programming_examples/llama3/multi_launch_builder/o_ffn_multi.py
programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py
programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py
programming_examples/llama3/multi_launch_builder/rms_gemv_rope_multi.py
```

(Note: 7 occurrences of `from llama3.kernel_builder.X import Y` confirmed via grep, all in the files listed above. Also 1 internal docstring-only reference at `kernel_builder/external_kernels.py:98` that doesn't need code change.)

**Created files (smoke-test artifact):**

```
programming_examples/llama32_1b_instruct/   # scaffolded by deploy-new-llm during smoke test
├── (full mirror of llama3/, with config swapped to instruct variant)
├── TODO.md
└── docs/development_progress/{progress.md, LESSONS.md, debug_log.md}
```

---

## Skill Authoring Convention

**Every SKILL.md follows this template** (per spec §8):

```markdown
---
name: <skill-name>
description: <one-line trigger description used by Claude to decide whether to invoke>
---

## Purpose
Single-paragraph what-and-when.

## Knowledge base references
Read these BEFORE acting if they apply:
- `programming_examples/llama3/docs/development_progress/<relevant-doc>.md`

## Workflow
Step-by-step procedure.

## Verification
How to know the gate/work-item passed.

## Failure modes
Common failures and which cross-cutting skill to invoke.

## Update protocol
What to write back to TODO.md / progress.md when done.
```

**Skill validation pattern** (replaces traditional unit-test TDD for markdown skills):

For each skill, before committing:
1. Walk through the workflow against a known scenario from LLAMA history (cited in the spec) and verify each step is concrete enough for a fresh Claude session to execute without ambiguity.
2. Confirm every `<path>` placeholder is real (file actually exists).
3. Confirm every "invoke skill: X" reference points to a skill that exists or is in this plan.
4. Confirm trigger pattern (for recipe skills) matches a real error string from `programming_examples/llama3/docs/development_progress/compiler_issues/*.md`.

---

## Task 1: Verify llama3 baseline before any changes

**Files:**
- None modified
- Read: `programming_examples/llama3/Makefile`

- [ ] **Step 1: Confirm clean working tree**

```bash
cd /home/jiajli/apps/mlir-air
git status --short
```

Expected: clean (no uncommitted changes besides the design spec/plan if not yet committed).

- [ ] **Step 2: Verify llama3 currently runs end-to-end**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make compile 2>&1 | tail -20
```

Expected: completes successfully; final line includes "kernels compiled and cached" or equivalent. Note timestamp of cache so we can detect regression.

- [ ] **Step 3: Run a quick inference smoke**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make run N_TOKENS=5 2>&1 | tail -10
```

Expected: token output appears (e.g., " Paris" or similar correct generation). No errors.

- [ ] **Step 4: Record baseline state**

Write the cache-dir mtime and a snippet of the run output into `/tmp/llama3_baseline.txt`:

```bash
ls -l /home/jiajli/apps/mlir-air/programming_examples/llama3/prefill_kernel_cache/manifest.json > /tmp/llama3_baseline.txt
echo "---" >> /tmp/llama3_baseline.txt
cd /home/jiajli/apps/mlir-air/programming_examples/llama3 && make run N_TOKENS=3 2>&1 | tail -5 >> /tmp/llama3_baseline.txt
cat /tmp/llama3_baseline.txt
```

Keep this file as the regression baseline for Task 3.

---

## Task 2: Lift `kernel_builder/` to `_llm_shared/` and rewrite imports

**Files:**
- Move: `programming_examples/llama3/kernel_builder/` → `programming_examples/_llm_shared/kernel_builder/`
- Modify: 7 Python files (see "Modified files" above) — replace `from llama3.kernel_builder.X import` with `from _llm_shared.kernel_builder.X import`

- [ ] **Step 1: Create `_llm_shared/` parent directory**

```bash
mkdir -p /home/jiajli/apps/mlir-air/programming_examples/_llm_shared
```

- [ ] **Step 2: Move kernel_builder via `git mv` (preserves history)**

```bash
cd /home/jiajli/apps/mlir-air
git mv programming_examples/llama3/kernel_builder programming_examples/_llm_shared/kernel_builder
git status --short
```

Expected: shows the moves as renames (R), not deletes+adds.

- [ ] **Step 3: Add `__init__.py` to `_llm_shared/` so it's a Python package**

Create `programming_examples/_llm_shared/__init__.py` with content:

```python
"""Shared infrastructure for LLM deployments on AMD NPU2.

Lifted from programming_examples/llama3/ on 2026-04-17 to enable reuse
across multiple LLM deployments. See README.md for usage.
"""
```

```bash
cd /home/jiajli/apps/mlir-air
cat > programming_examples/_llm_shared/__init__.py <<'EOF'
"""Shared infrastructure for LLM deployments on AMD NPU2.

Lifted from programming_examples/llama3/ on 2026-04-17 to enable reuse
across multiple LLM deployments. See README.md for usage.
"""
EOF
```

- [ ] **Step 4: Rewrite imports in `llama3_inference.py`**

Edit `/home/jiajli/apps/mlir-air/programming_examples/llama3/llama3_inference.py`:

Replace:
```python
from llama3.kernel_builder.cache import KernelCache, prepare_air_project
from llama3.kernel_builder.external_kernels import compile_all_external_kernels
```

With:
```python
from _llm_shared.kernel_builder.cache import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels
```

- [ ] **Step 5: Rewrite imports in `llama3_decode.py`**

Edit `/home/jiajli/apps/mlir-air/programming_examples/llama3/llama3_decode.py`:

Replace:
```python
from llama3.kernel_builder.cache import KernelCache, prepare_air_project
from llama3.kernel_builder.gemm_builder import _build_gemm_module
```

With:
```python
from _llm_shared.kernel_builder.cache import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module
```

- [ ] **Step 6: Rewrite imports in 6 multi_launch_builder files**

For each file in:
- `programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py`
- `programming_examples/llama3/multi_launch_builder/rms_gemv_rope_multi.py`
- `programming_examples/llama3/multi_launch_builder/o_ffn_multi.py`
- `programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py`
- `programming_examples/llama3/multi_launch_builder/lm_head_multi.py`
- `programming_examples/llama3/multi_launch_builder/lm_head_gemv_multi.py`

Replace `from llama3.kernel_builder.stitching import (` with `from _llm_shared.kernel_builder.stitching import (`.

(Each file has exactly one such import — confirmed via grep.)

- [ ] **Step 7: Confirm no stale `llama3.kernel_builder` references remain**

```bash
grep -rn "from llama3\.kernel_builder\|import llama3\.kernel_builder" /home/jiajli/apps/mlir-air/programming_examples/ 2>/dev/null
```

Expected: empty output. (The docstring-only reference at `_llm_shared/kernel_builder/external_kernels.py:98` saying "Compile silu_and_mul.o from kernel_builder/ffn_swiglu/silu_and_mul.cc" can stay — it's a comment about a relative path inside the package.)

- [ ] **Step 8: Verify llama3 still imports cleanly (without compiling/running)**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
import llama3_inference
import llama3_prefill
import llama3_decode
print('OK: all modules import cleanly')
"
```

Expected: `OK: all modules import cleanly`.

If any `ModuleNotFoundError`, find missing import and fix before continuing. Do not proceed.

- [ ] **Step 9: Commit the lift refactor as a separate commit**

```bash
cd /home/jiajli/apps/mlir-air
git add -A programming_examples/_llm_shared programming_examples/llama3
git commit -m "$(cat <<'EOF'
Lift kernel_builder to programming_examples/_llm_shared/

Single-source-of-truth refactor enabling reuse across LLM deployments.
Moved llama3/kernel_builder/ to _llm_shared/kernel_builder/ via git mv
(preserves history). Updated 8 callers (2 in llama3/, 6 in
multi_launch_builder/) from `llama3.kernel_builder` to `_llm_shared.kernel_builder`.

Per design spec docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md §3.
EOF
)"
git log -1 --oneline
```

Expected: commit succeeds; HEAD shows the lift commit.

---

## Task 3: Verify llama3 still works end-to-end after the lift

**Files:**
- None modified
- Compare: against `/tmp/llama3_baseline.txt` from Task 1

- [ ] **Step 1: Recompile to ensure refactor didn't break compilation**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make clean
make compile 2>&1 | tail -20
```

Expected: completes without errors. (Will recompile from scratch since `make clean` removed `build_peano/`. The kernel cache in `prefill_kernel_cache/` may persist.)

- [ ] **Step 2: Run inference and verify same output as baseline**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama3
make run N_TOKENS=3 2>&1 | tail -5 > /tmp/llama3_after_lift.txt
cat /tmp/llama3_after_lift.txt
echo "---BASELINE---"
tail -3 /tmp/llama3_baseline.txt
```

Expected: token output matches the baseline output (same predicted tokens for same prompt). If different — investigate before continuing.

- [ ] **Step 3: If anything regressed, halt the plan and resolve**

Do not proceed to Task 4 if Task 3 Step 2 shows regression. The lift is a load-bearing prerequisite for everything that follows.

(No commit in this task — verification only.)

---

## Task 4: Author `debug-bo-corruption` recipe skill

**Files:**
- Create: `.claude/skills/debug-bo-corruption/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/debug-bo-corruption
```

- [ ] **Step 2: Author SKILL.md**

Create `/home/jiajli/apps/mlir-air/.claude/skills/debug-bo-corruption/SKILL.md` with this content:

````markdown
---
name: debug-bo-corruption
description: Use when an NPU kernel passes its standalone shape test but produces NaN, garbage, or stale values when invoked as part of a larger pipeline. Common symptoms: correct first invocation but wrong on subsequent calls; correct in isolation but wrong when chained with other kernels.
---

## Purpose
Diagnose and auto-fix Buffer Object (BO) corruption issues in multi-launch and per-layer NPU pipelines. These bugs are characterized by a kernel that *individually* passes correctness but produces wrong output when integrated.

## Knowledge base references
Read these BEFORE acting if they apply:
- `programming_examples/llama3/docs/development_progress/multi-launch/host_optimization.md` — `static_input_indices`, `intermediate_indices`, BO write/read overhead
- `programming_examples/llama3/docs/development_progress/compiler_issues/herd_load_bug.md` — herd_load vs segment_load semantics

## Trigger pattern

This skill matches when ANY of these apply:
- Output tensor contains NaN despite kernel passing standalone XRTRunner test
- Kernel produces correct output on first invocation but wrong on second+ invocation in a loop
- Kernel produces correct output standalone but wrong when chained with another kernel
- Output tensor has correct shape but values match a *previous* layer/iteration

## Hypothesis tree

1. **Stale BO state from prior call** — `static_input_indices` not set, so a buffer that was written once is being re-read with stale values from a different invocation
2. **Intermediate buffer not marked** — `intermediate_indices` missing, so a buffer the kernel overwrites is being written from host with stale data each call
3. **Buffer aliasing across layers** — same BO object accidentally shared across two layers' per-layer caches
4. **Bare herd missing segment wrapper** — `airrt.herd_load` failing silently when herd is not wrapped in `air.segment` (see `compiler_issues/herd_load_bug.md` and the `project_multi_launch_segment` memory)

## Auto-fix attempts

For each hypothesis in order:

**(1) Stale BO state**
- Find the kernel's `cache.load_and_run(...)` call
- Check whether `static_input_indices=[...]` is passed
- Cross-reference with the kernel's IR: which input indices are weight buffers (written once) vs activation buffers (written every call)?
- Add missing weight indices to `static_input_indices`
- Re-test

**(2) Intermediate buffer not marked**
- Find buffers in the kernel that are *output* destinations the kernel will fully overwrite
- Check whether `intermediate_indices=[...]` lists these
- Add missing indices
- Re-test

**(3) Buffer aliasing across layers**
- Search for `bo_key` strings across layers
- Confirm each layer uses a unique `bo_key` (e.g., `f"kernel_L{layer_idx}"`)
- If same key reused across layers, fix by parameterizing on layer index

**(4) Bare herd missing segment wrapper**
- Search the multi-launch builder for `air.herd` ops
- Check each is wrapped in `air.segment`
- If not, wrap (see `kernel_builder/stitching.py` for `wrap_herd_in_segment` helper if present, else add manually)

## Verification

After applying any fix:
1. Re-run the failing test (the standalone XRT runner test or the integration test that surfaced the bug)
2. Confirm output matches CPU reference within `rtol=1e-3, atol=1e-5`
3. Run the test 3 times in a loop to confirm consistency across invocations

If fix succeeded: record `recovered_via=debug-bo-corruption` and which hypothesis fired in `<model>/docs/development_progress/debug_log.md`. Advance.

If no hypothesis fixed it: escalate. Update `<model>/TODO.md` "Active blockers" with the failing test, the hypotheses tried, and the unchanged failure output.

## Update protocol

On success: append to `<model>/docs/development_progress/debug_log.md`:
```
## debug-bo-corruption recovery (YYYY-MM-DD)
- Failing item: <kernel/test name>
- Hypothesis fired: (1) / (2) / (3) / (4)
- Fix applied: <one-line description>
- Verified: 3/3 consistent runs
```

On escalation: append to `<model>/TODO.md` Active blockers section with full failure context.
````

- [ ] **Step 3: Validate the skill mentally against a known scenario**

The LLAMA progress docs describe BO corruption discovered during Phase 4 (per `perf_optimization.md`). Walk through: would this skill, given the original symptom ("kernel passes standalone but corrupts on second call"), have triggered hypothesis (1) and led to the `static_input_indices` fix that the human eventually applied? If yes, the skill is well-grounded. If no, revise.

(No code changes — review only. If the skill needs revision, edit and re-validate.)

- [ ] **Step 4: Commit the skill**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/debug-bo-corruption/SKILL.md
git commit -m "$(cat <<'EOF'
Add debug-bo-corruption recipe skill

Cross-cutting recipe for BO corruption failures in multi-launch /
per-layer NPU pipelines. Covers 4 hypothesis categories with concrete
auto-fix steps. Cites llama3 docs for context.
EOF
)"
```

---

## Task 5: Author `debug-multi-launch-merge` recipe skill

**Files:**
- Create: `.claude/skills/debug-multi-launch-merge/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/debug-multi-launch-merge
```

Create `/home/jiajli/apps/mlir-air/.claude/skills/debug-multi-launch-merge/SKILL.md` with content:

````markdown
---
name: debug-multi-launch-merge
description: Use when attempting to merge multiple kernel launches into a single multi-launch ELF and the compile fails (BD exhaustion, channel routing, herd shape conflict, IR validation error). Recipe for diagnosing and recovering from kernel-fusion failures.
---

## Purpose
When stitching kernels together via `_llm_shared/kernel_builder/stitching.py` to produce a multi-launch ELF, the AIE compiler often rejects the merged module for hardware-resource reasons (BD count, channel count, herd shape, routing). This skill diagnoses which constraint was hit and applies the appropriate workaround.

## Knowledge base references
Read these BEFORE acting if they apply:
- `programming_examples/llama3/docs/development_progress/compiler_issues/multi_launch_blockers.md` — exhaustive list of merge constraints
- `programming_examples/llama3/docs/development_progress/compiler_issues/multi_launch_root_cause.md` — root cause analyses
- `programming_examples/llama3/docs/development_progress/multi-launch/full_block.md` — why attention cannot merge further
- `programming_examples/llama3/docs/development_progress/multi-launch/compiler_scaling.md` — compile time scaling with ELF size
- `programming_examples/llama3/docs/development_progress/compiler_issues/weight_broadcast_dma.md` — DMA stride limitation for BF16 broadcast (also: memory `project_bf16_dma_stride`)

## Trigger pattern

This skill matches when ANY of these appear in compile stderr:
- `buffer descriptor` / `BD` / `out of buffer descriptors` (BD exhaustion)
- `channel routing` / `cannot route` / `channel allocation failed`
- `herd shape mismatch` / `herd dimension`
- `aie.tile` location conflict
- `airrt.herd_load` not found / undefined symbol (see `debug-bo-corruption` for the segment-wrapper fix)
- `stride must be 1` for BF16 / sub-32b types (DMA constraint)

## Hypothesis tree

1. **BD exhaustion** — too many channels in the merged ELF; AIE2P has finite BD slots per tile
2. **Channel routing congestion** — adjacent launches use overlapping channel IDs that cannot coexist physically
3. **Herd shape conflict** — two launches require different herd shapes (e.g., [8,4] and [8,1]) and cannot coexist in one segment
4. **Bare herd missing segment** — see `debug-bo-corruption` hypothesis (4)
5. **DMA stride limitation** — sub-32b type with stride > 1 (see memory `project_bf16_dma_stride`)
6. **Compile-time blowup** — merge succeeded structurally but compile takes >5 min (per `compiler_scaling.md`)

## Auto-fix attempts

For each hypothesis in order:

**(1) BD exhaustion**
- Identify which kernels are being merged
- Un-merge the most recently added launch; keep the prior set
- Recompile and verify

**(2) Channel routing congestion**
- Renumber channels in one of the offending kernels (use `_llm_shared/kernel_builder/stitching.py` channel-rename helper)
- Recompile and verify

**(3) Herd shape conflict**
- The two launches cannot merge in this configuration
- Either: (a) re-tile one launch to match the other's herd shape, or (b) keep them as separate XRT calls
- Recommend (b) and skip the merge

**(4) Bare herd missing segment** — invoke `debug-bo-corruption` hypothesis (4)

**(5) DMA stride limitation**
- Restructure data layout so the offending DMA has stride=1
- Often requires changing memref shape or transpose order in the producer kernel
- See `compiler_issues/weight_broadcast_dma.md` for examples

**(6) Compile-time blowup**
- Not a correctness bug but a workflow blocker
- Reduce the merge scope (drop the most-recent launch from the merge set)
- Document in TODO.md as a soft cap

## Verification

After applying a fix:
1. Recompile the merged ELF — should succeed within reasonable time (<5 min)
2. Run the merged ELF on the same input as a single XRT call
3. Compare output against the unmerged baseline (same kernels run as separate XRT calls)
4. Output must match within `rtol=1e-3, atol=1e-5`

## Update protocol

Same pattern as `debug-bo-corruption`: log to `<model>/docs/development_progress/debug_log.md` on recovery; update `<model>/TODO.md` blockers on escalation.
````

- [ ] **Step 2: Validate against known scenario**

The LLAMA progress docs describe several multi-launch merge failures (BD exhaustion when trying to merge attention with surrounding kernels, herd shape conflict between [8,4] and [8,1], etc.). Walk through: would this skill have triggered the right hypothesis for each? Revise if gaps.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/debug-multi-launch-merge/SKILL.md
git commit -m "Add debug-multi-launch-merge recipe skill"
```

---

## Task 6: Author `merge-multi-launch-kernels` procedural skill

**Files:**
- Create: `.claude/skills/merge-multi-launch-kernels/SKILL.md`

- [ ] **Step 1: Create skill directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/merge-multi-launch-kernels
```

Create `/home/jiajli/apps/mlir-air/.claude/skills/merge-multi-launch-kernels/SKILL.md`:

````markdown
---
name: merge-multi-launch-kernels
description: Use when wanting to fuse multiple separate kernel launches into a single multi-launch ELF (single XRT invocation). The procedural recipe for the actual merge operation. Invoked by optimize-prefill-perf and optimize-decode-perf phase skills.
---

## Purpose
The procedure for merging N separate kernel launches into one multi-launch ELF using `_llm_shared/kernel_builder/stitching.py`. Reduces XRT dispatch overhead (each invocation costs ~50–200 µs on NPU2). For LLAMA: 10 launches/layer → 3 launches/layer was the major prefill perf win.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/llama3/docs/development_progress/multi-launch/host_optimization.md` — host-side BO write/read overhead, why merging helps
- `programming_examples/llama3/docs/development_progress/multi-launch/decode_merging.md` — decode-specific merge patterns (extern kernel rename for K=8192)
- `programming_examples/llama3/docs/development_progress/multi-launch/full_block.md` — what does NOT merge (attention)
- `programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py` — reference implementation of a 6-launch merge
- `programming_examples/llama3/multi_launch_builder/o_ffn_multi.py` — reference 8-launch merge

## Workflow

### Step 1: Identify merge candidates
List the per-layer kernel sequence. Mark each as one of:
- **Mergeable** — pure compute (RMSNorm, GEMM, GEMV, RoPE, eltwise)
- **Hard-stop** — has data-dependent control flow or unsupported merge pattern (FlashAttention, see `full_block.md`)
- **Conditional** — mergeable but must check herd-shape compatibility with neighbors

### Step 2: Pick the merge boundary
Group consecutive mergeable launches between hard-stops. Each group becomes one multi-launch ELF.

For a typical decoder-only LLM, the merge boundaries are:
- **Group A**: RMSNorm + Q/K/V GEMM/GEMV + RoPE_Q + RoPE_K (6 launches, like `rms_gemms_rope_multi.py`)
- **Hard-stop**: Attention (separate XRT call)
- **Group B**: O GEMM/GEMV + Add + RMSNorm + Gate + Up + SiLU+Mul + Down + Add (8 launches, like `o_ffn_multi.py`)

### Step 3: Author the multi-launch builder

For each group, create a Python file in `<model>/multi_launch_builder/<group_name>_multi.py` that:

1. Builds each sub-kernel's IR via `@module_builder`
2. Imports stitching helpers: `from _llm_shared.kernel_builder.stitching import (rename_ssa_with_prefix, fix_launch_func_args, wrap_herd_in_segment, ...)`
3. For each sub-kernel: extract its function body
4. Rename SSA values with a per-kernel prefix to avoid collisions
5. Remap function arguments to the merged module's args
6. Concatenate the renamed bodies into a single `air.launch` (or `air.segment` if any sub-kernel uses a bare herd)
7. Return the merged MLIR module

Use `programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py` as the canonical reference.

### Step 4: Compile the merged ELF

Use `KernelCache.compile_and_cache(name=<group_name>, builder=<your_builder_function>, ...)`.

If compile fails → invoke `debug-multi-launch-merge` recipe skill.

### Step 5: Validate against unmerged baseline

Run the merged ELF on a fixed input. Compare against running the un-merged kernels as separate XRT calls. Output must match within `rtol=1e-3, atol=1e-5`.

### Step 6: Measure perf gain

Time the merged version vs unmerged. Expect ≥20% reduction per merged group at NPU2 dispatch overhead levels (LLAMA observed larger gains for tighter merges). Record in `<model>/docs/development_progress/progress.md`.

## Verification

A merge is "successful" when:
1. Merged ELF compiles without errors
2. Output matches unmerged baseline within tolerance
3. Wall-clock time is lower than unmerged

If (1) fails → invoke `debug-multi-launch-merge`.
If (2) fails → bisect: drop the most-recent merge addition, re-validate; if still wrong, the bug is in the *first* merge.
If (3) fails (compile slower than unmerged) → not a correctness bug; document and either accept or revert.

## Failure modes
- Compile failure → `debug-multi-launch-merge`
- Output mismatch → bisect by un-merging
- BO corruption after merge → `debug-bo-corruption`

## Update protocol
Append to `<model>/docs/development_progress/progress.md`:
```
## Multi-launch merge: <group_name>
- Sub-kernels merged: N
- Latency before: X ms
- Latency after: Y ms
- Gain: Z%
```
````

- [ ] **Step 2: Validate against the LLAMA history**

The LLAMA Phase 4 documents describe exactly this merge process for `rms_gemms_rope` (6 launches) and `o_ffn` (8 launches). Walk through this skill applied to one of those — does it produce the same procedure the human/Claude followed? Revise if gaps.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/merge-multi-launch-kernels/SKILL.md
git commit -m "Add merge-multi-launch-kernels procedural skill"
```

---

## Task 7: Author `bootstrap-model-config` skill (Phase 0)

**Files:**
- Create: `.claude/skills/bootstrap-model-config/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/bootstrap-model-config
```

Create `.claude/skills/bootstrap-model-config/SKILL.md`:

````markdown
---
name: bootstrap-model-config
description: Phase 0 of LLM deployment — adapt config dataclass and HuggingFace weight loader for the target model. Produce <model>_weights.py and <model>_reference.py. Invoked by deploy-new-llm after scaffolding.
---

## Purpose
Translate a HuggingFace model into the data structures the rest of the pipeline expects: a `Config` dataclass with NPU-relevant fields, a weight loader that produces correctly-shaped numpy arrays, and a CPU F32 reference implementation for downstream correctness gates.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/llama3/llama3_weights.py` — reference Config dataclass + HF weight loading
- `programming_examples/llama3/llama3_reference.py` — reference CPU F32 inference
- `programming_examples/llama3/docs/explain.md` — RoPE half-split convention details

## Workflow

### Step 1: Read the HF config
Fetch `config.json` for the target model from HuggingFace. Extract:
- `num_hidden_layers` → `n_layers`
- `hidden_size` → `emb_dim`
- `num_attention_heads` → `n_heads`
- `num_key_value_heads` → `n_kv_heads` (default to `n_heads` if absent → MHA)
- `intermediate_size` → `hidden_dim`
- `vocab_size` → `vocab_size`
- `rope_theta` → `rope_base` (default 10000.0 if absent)
- `head_dim` (compute as `emb_dim // n_heads` if absent)

### Step 2: Architecture compatibility check
Before writing any file, confirm the model is in-scope (per spec §2):
- Architecture must be in `["LlamaForCausalLM", "MistralForCausalLM" (only if no sliding window), "Qwen2ForCausalLM" (only if no QKV bias), ...]` — i.e., a decoder-only with RMSNorm + SwiGLU + RoPE + GQA/MHA
- Reject if: MoE layers present, sliding-window attention, MLA, QKV bias (without bias support)
- Reject explicitly with a clear message; do NOT scaffold a model the rest of the pipeline can't handle

### Step 3: Generate `<model>_weights.py`
Copy `programming_examples/llama3/llama3_weights.py` to `<model>/<model>_weights.py`. Modify:
- `LlamaConfig` dataclass defaults → match Step 1 values
- HF weight name remapping in `load_weights()` — most LLAMA-derived models share the same names (`model.layers.<i>.self_attn.q_proj.weight` etc.); confirm via inspecting the safetensors index. If different, write an explicit mapping.
- `generate_rope_lut()` — verify `rope_base` is parameterized and uses the new value

### Step 4: Generate `<model>_reference.py`
Copy `programming_examples/llama3/llama3_reference.py` to `<model>/<model>_reference.py`. Modify:
- Imports: change `from llama3_weights import` to `from <model>_weights import`
- Config references: ensure all hardcoded shapes are replaced by config attributes
- For unfamiliar architectures (different attention masking, etc.) — adapt carefully

### Step 5: Smoke-test the reference
Run the CPU reference on a canonical prompt:

```bash
cd programming_examples/<model>
python3 <model>_reference.py --prompt "The capital of France is" --n-tokens 5
```

Expected: produces lexically sensible token output.

## Verification (Phase 0 gate)

Phase 0 PASSES when ALL true:
1. `<model>_weights.py` loads all expected tensors with shapes matching the config (programmatic check: every layer index 0..n_layers-1 has q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, attn_norm, ffn_norm)
2. `<model>_reference.py` runs and produces stable output (no NaN, sensible logit distribution: top-5 logits within reasonable range, not all uniform)
3. Reference output for canonical prompt is lexically sensible (human eyeball check OR pre-computed expected from HuggingFace transformers library)

## Failure modes
- Architecture rejected at Step 2 → escalate, deployment cannot proceed
- Weight name mismatch at Step 3 → human resolves (variant naming conventions)
- Reference produces NaN → likely a config error (wrong head_dim, wrong rope_theta); revisit Step 1
- Reference produces gibberish → likely a weight-load shape mismatch; check `np.ascontiguousarray()` and tensor dtypes

## Update protocol

On Phase 0 PASS, append to `<model>/docs/development_progress/progress.md`:
```
## Phase 0: Bootstrap (PASSED YYYY-MM-DD)
- HF model: <id>
- Config: n_layers=N, emb_dim=D, n_heads=H, n_kv_heads=K, hidden_dim=F, vocab=V, rope_base=R
- Reference smoke output: "<first 5 tokens>"
```

Update `<model>/TODO.md`: mark Phase 0 checkbox; populate "Resolved config" section.
````

- [ ] **Step 2: Validate against LLAMA history**

`programming_examples/llama3/llama3_weights.py` IS the reference. Walk through this skill against an imaginary new LLM (TinyLlama-1.1B). Does it produce a correct `tinyllama_weights.py`? Specifically: would Step 2 (compat check) correctly reject Qwen2 with QKV bias? Revise if gaps.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/bootstrap-model-config/SKILL.md
git commit -m "Add bootstrap-model-config skill (Phase 0)"
```

---

## Task 8: Author `validate-per-kernel-shapes` skill (Phase 1)

**Files:**
- Create: `.claude/skills/validate-per-kernel-shapes/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/validate-per-kernel-shapes
```

Create `.claude/skills/validate-per-kernel-shapes/SKILL.md`:

````markdown
---
name: validate-per-kernel-shapes
description: Phase 1 of LLM deployment — verify each unique kernel shape the model needs passes against CPU reference. Invoked by deploy-new-llm after Phase 0 gate passes.
---

## Purpose
Before integration, prove each individual kernel (RMSNorm, GEMM, GEMV, RoPE, FlashAttention, SwiGLU, eltwise add) works correctly on NPU2 at every shape the new model needs. This isolates per-shape failures from integration bugs.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/llama3/docs/development_progress/kernels/gemm.md` — GEMM tile config strategy
- `programming_examples/llama3/docs/development_progress/kernels/gemv.md` — GEMV herd layouts (8×1 K=2048; extern rename for K=8192)
- `programming_examples/llama3/docs/development_progress/kernels/rmsnorm.md` — 8-tile broadcast strategy
- `programming_examples/llama3/docs/development_progress/kernels/rope.md` — RoPE LUT layout, half-split convention
- `programming_examples/llama3/docs/development_progress/kernels/flash_attention.md` — seq-first layout, causal masking
- `programming_examples/llama3/docs/development_progress/kernels/ffn_swiglu.md`, `silu_and_mul.md`
- `programming_examples/llama3/docs/development_progress/kernels/eltwise_add.md`

## Workflow

### Step 1: Enumerate the unique shapes from `<model>_weights.py` Config
Compute the set of (kernel_type, shape_tuple) pairs the model needs:

```python
shapes = []
# RMSNorm: per-block input
shapes.append(("rmsnorm", (config.emb_dim,)))
# GEMM: prefill (M=seq_len, N, K=emb_dim) for Q/K/V/O/Gate/Up/Down
for (m, n, k) in [
    (config.seq_len, config.n_heads * config.head_dim, config.emb_dim),  # Q
    (config.seq_len, config.n_kv_heads * config.head_dim, config.emb_dim),  # K
    (config.seq_len, config.n_kv_heads * config.head_dim, config.emb_dim),  # V
    (config.seq_len, config.emb_dim, config.n_heads * config.head_dim),  # O
    (config.seq_len, config.hidden_dim, config.emb_dim),  # Gate
    (config.seq_len, config.hidden_dim, config.emb_dim),  # Up
    (config.seq_len, config.emb_dim, config.hidden_dim),  # Down
    (config.seq_len, config.vocab_size, config.emb_dim),  # LM head
]:
    shapes.append(("gemm", (m, n, k)))
# GEMV: decode (M=1) versions of the above
# RoPE: head_dim
# FlashAttention: (seq_len, n_heads, head_dim, n_kv_heads)
# SwiGLU/SiLU+Mul: hidden_dim
# Eltwise add: emb_dim
```

Deduplicate. Most LLAMA-derived models produce 12–16 unique shape tuples.

### Step 2: Loop over the inner debug-loop pattern (per spec §7)

For each `(kernel_type, shape)`:

```
1. Build the kernel module via _llm_shared/kernel_builder/<builder>
2. Use XRTRunner.run_test(...) with random inputs and CPU reference outputs
3. If passed → record pass, advance
4. If failed:
   a. Match error against debug-bo-corruption trigger pattern → if match, invoke that recipe
   b. Else invoke superpowers:systematic-debugging
   c. If still failed → escalate
```

Bound: 1 retry per recipe per shape; 1 systematic-debugging attempt per shape.

### Step 3: Produce a pass/fail table
Write to `<model>/docs/development_progress/phase1_kernel_shapes.md`:

```
| Kernel | Shape | Status | Recovered via |
|--------|-------|--------|---------------|
| RMSNorm | (2048,) | PASS | — |
| GEMM | (2048, 2048, 2048) | PASS | — |
| GEMM | (2048, 5632, 2048) | PASS_RECOVERED | debug-bo-corruption (1) |
| ...
```

## Verification (Phase 1 gate)

Phase 1 PASSES when ALL shapes show `PASS` or `PASS_RECOVERED` in the table. Any `FAIL` blocks the gate.

## Failure modes
- Shape `FAIL` after recipe + systematic-debugging → escalate to human via TODO.md "Active blockers"
- Compilation hang (>10 min) → likely a compiler-scaling issue; cap and document

## Update protocol

On Phase 1 PASS:
- Update `<model>/TODO.md`: mark Phase 1, append "(N/N PASSED)"
- Append phase1_kernel_shapes.md content (or summary) to `<model>/docs/development_progress/progress.md`
````

- [ ] **Step 2: Validate against LLAMA history**

LLAMA Phase 1 in `progress.md` describes validating 9 kernels at LLAMA-3.2-1B shapes. Does this skill, applied to LLAMA, reproduce that workflow? Revise if gaps.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/validate-per-kernel-shapes/SKILL.md
git commit -m "Add validate-per-kernel-shapes skill (Phase 1)"
```

---

## Task 9: Author `integrate-single-block` skill (Phase 2)

**Files:**
- Create: `.claude/skills/integrate-single-block/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/integrate-single-block
```

Create `.claude/skills/integrate-single-block/SKILL.md`:

````markdown
---
name: integrate-single-block
description: Phase 2 of LLM deployment — assemble one transformer block on NPU and verify cosine similarity > 0.99 against CPU reference. Invoked by deploy-new-llm after Phase 1 gate.
---

## Purpose
Once individual kernels work, integrate them into a single transformer block. This catches integration bugs (wrong tensor layouts between kernels, missing transposes, intermediate type mismatches) before scaling to N layers.

## Knowledge base references
- `programming_examples/llama3/llama3_prefill.py:run_transformer_block` — reference single-block pipeline
- `programming_examples/llama3/docs/development_progress/progress.md` — LLAMA Phase 2 log (CPU fallback strategy)
- `programming_examples/llama3/docs/explain.md` — kernel directory map

## Workflow

### Step 1: Wire one block with all-NPU kernels (no CPU fallback initially)
In `<model>/<model>_prefill.py`, implement `run_single_block(layer_idx, hidden, weights, ...)` that:
1. RMSNorm hidden → norm_out
2. Q/K/V GEMM on norm_out using layer weights
3. RoPE on Q and K
4. FlashAttention(Q, K, V) → attn_out
5. O GEMM on attn_out
6. Residual add: hidden + o_out → res1
7. RMSNorm res1 → norm2
8. Gate/Up GEMM on norm2
9. SiLU(Gate) * Up → swiglu_out
10. Down GEMM on swiglu_out
11. Residual add: res1 + down_out → block_out

### Step 2: Define correlation metric
```python
def cosine_sim(a, b):
    a_flat, b_flat = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def mae(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
```

### Step 3: Run NPU vs reference for layer 0
Pick a fixed input (e.g., embeddings of "The capital of France is"). Run both:
- NPU single block at layer 0 → npu_out
- `<model>_reference.py` single block at layer 0 → ref_out

Compute `cosine_sim(npu_out, ref_out)` and `mae(npu_out, ref_out)`.

### Step 4: If correlation < 0.99, bisect with CPU fallback
Replace each kernel one at a time with its CPU reference equivalent (using ops from `<model>_reference.py`). Find the kernel after which correlation drops below 0.99. That kernel is the offender — invoke `superpowers:systematic-debugging` on it.

### Step 5: Document
Record correlation table per kernel-bisect step in `<model>/docs/development_progress/phase2_block.md`.

## Verification (Phase 2 gate)

Phase 2 PASSES when:
- `cosine_sim(npu_block_out, ref_block_out) > 0.99` for layer 0
- `mae(npu_block_out, ref_block_out) < 1e-2`
- No NaN in NPU output

## Failure modes
- Correlation drops at Q/K/V GEMM → likely weight loading or tensor layout (check seq-first vs heads-first)
- Correlation drops at FlashAttention → likely causal masking missing or wrong (see `kernels/flash_attention.md`)
- Correlation drops at Down GEMM → BF16 truncation; check F32 accumulator (LLAMA Phase 3 fix per `progress.md`)
- NaN in output → invoke `debug-bo-corruption`

## Update protocol

On Phase 2 PASS:
- Append phase2 results to `progress.md`
- Update `TODO.md` Phase 2 checkbox
````

- [ ] **Step 2: Validate against LLAMA Phase 2 history**

LLAMA's Phase 2 hit specific bugs: non-contiguous weight arrays, wrong output index. Does this skill's bisect strategy localize those? Revise if gaps.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/integrate-single-block/SKILL.md
git commit -m "Add integrate-single-block skill (Phase 2)"
```

---

## Task 10: Author `validate-full-model-correctness` skill (Phase 3)

**Files:**
- Create: `.claude/skills/validate-full-model-correctness/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/validate-full-model-correctness
```

Create `.claude/skills/validate-full-model-correctness/SKILL.md`:

````markdown
---
name: validate-full-model-correctness
description: Phase 3 of LLM deployment — wire all N layers and verify top-1 prediction matches CPU reference for canonical prompts; per-layer correlation > 0.95 throughout. Invoked after Phase 2 gate.
---

## Purpose
Scale single-block correctness to the full N-layer stack. Catches accumulated drift across layers, KV cache bugs, layer-specific weight loading errors, and BF16 truncation that compounds.

## Knowledge base references
- `programming_examples/llama3/llama3_inference.py:run_npu_prefill` — reference full-stack pipeline
- `programming_examples/llama3/docs/development_progress/progress.md` — LLAMA Phase 3 (Paris test, F32 accumulator fix)

## Workflow

### Step 1: Wire all N layers
Implement `run_full_prefill(input_ids, weights, config)` that loops `run_single_block(layer_idx, ...)` for `layer_idx in range(config.n_layers)`, then applies final RMSNorm + LM head.

### Step 2: Define canonical prompts
At minimum 3 prompts that have unambiguous next-token completions in the reference model:
- "The capital of France is" (expected: " Paris" or similar)
- "1 + 1 = " (expected: "2")
- "The sky is" (expected: " blue" or similar)

(For instruct/chat models, use chat-templated equivalents.)

### Step 3: Run NPU vs reference for each prompt
Predict next token via:
- NPU full prefill → top-1 token
- CPU reference full prefill → top-1 token

### Step 4: If top-1 mismatch, run per-layer correlation analysis
For each layer 0..N-1:
- Compute `cosine_sim(npu_layer_out, ref_layer_out)`
- Identify the first layer where correlation drops below 0.95

That's the offending layer — bisect within the layer using Phase 2's per-kernel approach.

### Step 5: Document
Record per-layer correlation table in `<model>/docs/development_progress/phase3_full.md`.

## Verification (Phase 3 gate)

Phase 3 PASSES when:
- Top-1 NPU prediction matches CPU reference for ≥3 of 3 canonical prompts
- Per-layer correlation > 0.95 for all layers (no drift)
- No NaN anywhere in the stack

## Failure modes
- Top-1 mismatch but per-layer corr stays > 0.95 → likely LM head precision; invoke F32 accumulator pattern from `kernels/gemm.md`
- Per-layer corr drops at a single layer → likely a layer-indexed weight loading bug (check `bo_key=f"kernel_L{i}"`)
- Drift starts at layer 0 and worsens → likely BF16 accumulator issue throughout; F32 accumulator fix per LLAMA Phase 3

## Update protocol

On Phase 3 PASS, this is the **end-to-end correctness milestone**. Update `progress.md` with full results table, mark `TODO.md` Phase 3, advance to perf phases.
````

- [ ] **Step 2: Validate**

LLAMA Phase 3 documented 0.34 correlation issue (FlashAttention without causal mask) and BF16 truncation (Down GEMM 0.976 corr → F32 fix). Does this skill's bisect localize those? Yes for the per-layer drift case (would identify the layer with attention or down-GEMM at fault). For the "all-layers drift" BF16 case — would the per-layer corr table show monotonic degradation? Yes. OK, skill is sound.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/validate-full-model-correctness/SKILL.md
git commit -m "Add validate-full-model-correctness skill (Phase 3)"
```

---

## Task 11: Author `optimize-prefill-perf` skill (Phase 4)

**Files:**
- Create: `.claude/skills/optimize-prefill-perf/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/optimize-prefill-perf
```

Create `.claude/skills/optimize-prefill-perf/SKILL.md`:

````markdown
---
name: optimize-prefill-perf
description: Phase 4 of LLM deployment — apply the 5 known prefill optimization patterns from LLAMA Phase 4 (multi-launch merging, BO pre-loading, intermediate buffer reuse, seq-first layout, CPU→NPU op promotion). Invoked after Phase 3 correctness gate.
---

## Purpose
Apply the prefill optimization patterns that took LLAMA from 18.67s → 1.30s. Each pattern is mechanically applicable; the skill attempts each and records the gain or skip-reason. Judgment-heavy choices defer to human.

## Knowledge base references
- `programming_examples/llama3/docs/development_progress/perf_optimization.md` — full optimization journey
- `programming_examples/llama3/docs/development_progress/multi-launch/host_optimization.md` — `static_input_indices`, `intermediate_indices`
- `programming_examples/llama3/docs/development_progress/multi-launch/decode_merging.md` — extern kernel rename pattern (used here for shape-collision avoidance)

## Workflow

Apply each pattern in order. After each, re-run Phase 3 gate as regression check. If correctness regresses, revert the pattern.

### Pattern 1: Multi-launch merging
Invoke `merge-multi-launch-kernels` skill for the prefill kernel groups (Group A: rms+gemms+rope, Group B: o+ffn).

Expected gain: 2–4× reduction in XRT call count → measurable wall-clock reduction.

### Pattern 2: Per-layer Buffer Object pre-loading
Identify weight tensors that are written once and reused on every layer call. In the host runtime:
- Allocate per-layer BOs using `bo_key=f"kernel_L{layer_idx}"`
- Write weights to BOs during a `prepare_runtime()` setup phase (not on every call)
- Pass `static_input_indices=[<weight_indices>]` on every `cache.load_and_run()` to skip re-write

Expected gain: significant reduction in host-side BO write overhead (LLAMA observed multi-second prefill time savings).

### Pattern 3: Intermediate buffer reuse
For buffers the kernel fully overwrites (its outputs and scratch):
- Pass `intermediate_indices=[<output_indices>]` on `cache.load_and_run()` to skip the initial host write

Expected gain: smaller per-call host overhead.

### Pattern 4: Seq-first activation layout
If the model uses heads-first layout `(heads, seq, dim)` and the kernels can accept seq-first `(seq, heads*dim)` natively (via stride args), eliminate host transposes between kernels.

Specific conversion:
- RoPE: accept seq-first input
- FlashAttention: accept seq-first Q, K, V

Expected gain: eliminates 1–4 host-side transposes per layer.

### Pattern 5: CPU→NPU op promotion
If the current pipeline has CPU fallbacks for any small ops (eltwise add, RMSNorm, etc.), move them to NPU using existing kernels. Re-run Phase 3 gate to confirm correctness.

Expected gain: removes CPU/NPU sync overhead.

### Bookkeeping
For each pattern:
- If applied: record latency before/after, gain %, in `<model>/docs/development_progress/phase4_prefill.md`
- If skipped: record reason (e.g., "Pattern 4 N/A — model already uses seq-first by default")
- If failed: invoke relevant debug skill; if unrecoverable, log as known limitation, advance

## Verification (Phase 4 gate)

Phase 4 PASSES when:
- Prefill end-to-end latency measured and recorded
- ≥3 of 5 patterns successfully applied (or N/A with documented reason)
- No correctness regression (Phase 3 gate re-run is still PASS)

## Failure modes
- Multi-launch merge fails → `debug-multi-launch-merge`
- Output corruption after BO pre-loading → `debug-bo-corruption`
- Correctness regresses → revert the pattern, document reason

## Update protocol

Append to `progress.md` per-pattern table with measured gains. Update `TODO.md` Phase 4.
````

- [ ] **Step 2: Validate against LLAMA Phase 4**

Cross-check: each of the 5 patterns is documented in `programming_examples/llama3/docs/development_progress/perf_optimization.md`. Skill is grounded.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/optimize-prefill-perf/SKILL.md
git commit -m "Add optimize-prefill-perf skill (Phase 4)"
```

---

## Task 12: Author `optimize-decode-perf` skill (Phase 5)

**Files:**
- Create: `.claude/skills/optimize-decode-perf/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/optimize-decode-perf
```

Create `.claude/skills/optimize-decode-perf/SKILL.md`:

````markdown
---
name: optimize-decode-perf
description: Phase 5 of LLM deployment — apply the 5 known decode optimization patterns from LLAMA Phase 5 (multi-launch merging, static weight BOs, NPU LM Head GEMV, extern kernel rename, CPU→NPU promotion). Invoked after Phase 4 gate.
---

## Purpose
Apply the decode optimization patterns that took LLAMA from ~500ms/token → 92ms/token. Same structure as Phase 4 but tuned for the M=1 (single-token) case.

## Knowledge base references
- `programming_examples/llama3/docs/development_progress/multi-launch/decode_merging.md` — decode-specific merge patterns
- `programming_examples/llama3/docs/development_progress/decode_archive/DECODE_PROGRESS.md` — decode milestones
- `programming_examples/llama3/docs/development_progress/decode_archive/gemv_investigation.md` — AIR vs IRON GEMV perf

## Workflow

### Pattern 1: Multi-launch merging (decode variants)
Invoke `merge-multi-launch-kernels` for decode groups (rms+gemvs+rope, o+ffn). Same procedure as prefill but with GEMV instead of GEMM.

Expected: 10 launches/layer/token → 2–3 launches/layer/token.

### Pattern 2: Static weight BOs
Decode reuses every weight on every token. Convert weight BOs to allocated-once with `bo.map()` zero-copy access.

Expected gain: removes per-token BO write of all weights.

### Pattern 3: NPU LM Head GEMV (vocab projection)
Replace CPU LM head with NPU GEMV partitioned across vocab (LLAMA used 8-partition with `mv_k8192.o` extern rename). Each partition handles `vocab/8` rows × `emb_dim` columns.

Expected gain: LLAMA observed ~250ms → ~14ms.

### Pattern 4: Extern kernel rename (shape-collision avoidance)
If two GEMV shapes coexist in one ELF (e.g., K=2048 and K=8192) and need different kernel implementations, compile with `-D` symbol renames so they can be linked together.

See `_llm_shared/kernel_builder/external_kernels.py` and `programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py` for the existing K=8192 rename pattern.

### Pattern 5: CPU→NPU op promotion
Same as Phase 4 Pattern 5 but for decode-specific ops (typically the small attention step on a single query, if currently on CPU).

### Bookkeeping
Same as Phase 4: record per-pattern latency in `<model>/docs/development_progress/phase5_decode.md`.

## Verification (Phase 5 gate)

Phase 5 PASSES when:
- Decode latency measured (ms/token)
- ≥3 of 5 patterns applied or N/A
- No correctness regression (Phase 3 re-run still PASS)

## Failure modes
Same as Phase 4 plus:
- Extern kernel rename collision → check `-D` symbol mapping uniqueness; invoke `debug-multi-launch-merge`

## Update protocol
Append to `progress.md`. Update `TODO.md` Phase 5.
````

- [ ] **Step 2: Validate against LLAMA Phase 5**

Each pattern documented in LLAMA decode docs. Grounded.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/optimize-decode-perf/SKILL.md
git commit -m "Add optimize-decode-perf skill (Phase 5)"
```

---

## Task 13: Author `finalize-deployment` skill (Phase 6)

**Files:**
- Create: `.claude/skills/finalize-deployment/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/finalize-deployment
```

Create `.claude/skills/finalize-deployment/SKILL.md`:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/finalize-deployment/SKILL.md
git commit -m "Add finalize-deployment skill (Phase 6)"
```

---

## Task 14: Author `deploy-new-llm` entry skill

**Files:**
- Create: `.claude/skills/deploy-new-llm/SKILL.md`

- [ ] **Step 1: Create directory and SKILL.md**

```bash
mkdir -p /home/jiajli/apps/mlir-air/.claude/skills/deploy-new-llm
```

Create `.claude/skills/deploy-new-llm/SKILL.md`:

````markdown
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
````

- [ ] **Step 2: Validate**

Walk the entry skill against the smoke-test scenario (Llama-3.2-1B-Instruct, Task 15). Does it produce a clean scaffold and dispatch correctly? Yes — same arch as base, all phases should sail through.

- [ ] **Step 3: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add .claude/skills/deploy-new-llm/SKILL.md
git commit -m "Add deploy-new-llm entry skill"
```

---

## Task 15: Smoke test — deploy Llama-3.2-1B-Instruct end-to-end via the skills

**Goal:** Prove the skill chain runs cleanly on a trivial-delta target. Llama-3.2-1B-Instruct shares LLAMA-3.2-1B's exact architecture, differing only in fine-tuned weights. So all 7 phases should pass without any novel debug recipes firing.

**Files:**
- Created by skills: `programming_examples/llama32_1b_instruct/` (full scaffold)
- Created by skills: `programming_examples/llama32_1b_instruct/docs/development_progress/{progress.md, LESSONS.md, debug_log.md, phase*.md}`
- Created by skills: `programming_examples/llama32_1b_instruct/TODO.md`

- [ ] **Step 1: Pre-flight check — confirm all skills are present**

```bash
ls /home/jiajli/apps/mlir-air/.claude/skills/
```

Expected: 11 directories listed (deploy-new-llm, bootstrap-model-config, validate-per-kernel-shapes, integrate-single-block, validate-full-model-correctness, optimize-prefill-perf, optimize-decode-perf, finalize-deployment, debug-bo-corruption, debug-multi-launch-merge, merge-multi-launch-kernels).

- [ ] **Step 2: Confirm Llama-3.2-1B-Instruct weights are accessible**

(The user already supports this model — see commit `b9f2cd18`'s "instruct model support". Confirm weights are downloaded.)

- [ ] **Step 3: Invoke entry skill**

In a fresh Claude Code session, invoke:
```
/deploy-new-llm meta-llama/Llama-3.2-1B-Instruct --name llama32_1b_instruct
```

Or, if invoking from this same session, use the Skill tool to invoke `deploy-new-llm` with the same args.

- [ ] **Step 4: Walk through phases 0–6, approving each gate**

Each phase should PASS without escalation. Approve gate-by-gate.

Phase 0: Config + reference run, smoke output.
Phase 1: All shapes pass (same as base — should be no-op compared to llama3 cache).
Phase 2: Single block correlation > 0.99.
Phase 3: Top-1 prediction matches reference for 3 prompts.
Phase 4: 5/5 prefill patterns applied (since llama3 already has them all).
Phase 5: 5/5 decode patterns applied.
Phase 6: Final perf report.

- [ ] **Step 5: Verify the deployment artifact**

```bash
ls /home/jiajli/apps/mlir-air/programming_examples/llama32_1b_instruct/docs/development_progress/
cat /home/jiajli/apps/mlir-air/programming_examples/llama32_1b_instruct/docs/development_progress/progress.md
```

Expected: progress.md shows all 7 phases PASSED with measured perf numbers within 5% of LLAMA base (since arch is identical).

- [ ] **Step 6: Run final inference to confirm**

```bash
cd /home/jiajli/apps/mlir-air/programming_examples/llama32_1b_instruct
make run N_TOKENS=10 2>&1 | tail -5
```

Expected: produces sensible Llama-3.2-1B-Instruct chat-style output.

- [ ] **Step 7: If smoke test passed, capture lessons; if it failed, iterate skills**

If PASS: commit the new programming_examples directory + any skill refinements made during the test:
```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/llama32_1b_instruct .claude/skills
git commit -m "Smoke test: deploy llama32_1b_instruct via skill chain"
```

If FAIL at any phase: this is the *intended* discovery work of the smoke test. Identify the friction point (skill ambiguity, missing recipe, scaffolding bug), refine the relevant skill(s), commit refinements with messages like "refine: <skill> handles <case>", re-invoke from the failing phase.

---

## Task 16: Author `programming_examples/_llm_shared/README.md`

**Files:**
- Create: `programming_examples/_llm_shared/README.md`

- [ ] **Step 1: Write the README**

```bash
cat > /home/jiajli/apps/mlir-air/programming_examples/_llm_shared/README.md <<'EOF'
# `_llm_shared/` — Shared Infrastructure for LLM Deployments on AMD NPU2

This directory holds reusable infrastructure extracted from the LLAMA-3.2-1B
deployment (`programming_examples/llama3/`). It is consumed by per-model
deployments scaffolded via the `deploy-new-llm` skill.

## Contents

- `kernel_builder/` — Generic kernel-building infrastructure
  - `cache.py` — `KernelCache` (compile-once / run-many with manifest), `Profiler`
  - `stitching.py` — Text-based MLIR merging utilities (used by multi-launch builders)
  - `gemm_builder.py` — Generic GEMM transform IR
  - `external_kernels.py` — C++ kernel compilation pipeline
  - `rope_halfsplit.cc` — Half-split RoPE convention C++ kernel
  - `ffn_swiglu/` — Standalone SwiGLU AIR kernel + C++

## Usage

Per-model deployments import like:

```python
from _llm_shared.kernel_builder.cache import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.stitching import (
    rename_ssa_with_prefix,
    fix_launch_func_args,
    wrap_herd_in_segment,
)
from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module
```

This requires `programming_examples/` to be on `PYTHONPATH` (the existing
`sys.path.insert(0, "..")` pattern in `llama3_*.py` continues to work).

## How to deploy a new LLM using these utilities

Use the `deploy-new-llm` skill in `.claude/skills/`. It scaffolds a new
per-model directory and walks through the 7-phase workflow.

## History

Lifted from `programming_examples/llama3/kernel_builder/` on 2026-04-17 as
part of the LLM mapping skills initiative. See:
- `docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md`
- `docs/superpowers/plans/2026-04-17-llm-mapping-skills.md`

## Scope

The kernels and patterns here assume **decoder-only LLMs** with:
- Attention: GQA or MHA
- Normalization: RMSNorm
- FFN: SwiGLU
- Position encoding: RoPE (half-split convention)

MoE, sliding-window attention, MLA, and encoder-decoder architectures are
explicitly out of scope for this infrastructure.
EOF
```

- [ ] **Step 2: Commit**

```bash
cd /home/jiajli/apps/mlir-air
git add programming_examples/_llm_shared/README.md
git commit -m "Document _llm_shared/ usage for downstream deployments"
```

---

## Task 17: Final plan-level commit and tag

- [ ] **Step 1: Confirm all artifacts exist**

```bash
cd /home/jiajli/apps/mlir-air
echo "Skills:"
ls .claude/skills/
echo "---Shared infra:---"
ls programming_examples/_llm_shared/
echo "---Smoke test artifact:---"
ls programming_examples/llama32_1b_instruct/ 2>/dev/null || echo "(skipped if smoke test deferred)"
```

Expected: 11 skills, `_llm_shared/` populated with kernel_builder + README, smoke-test directory present (or noted as deferred).

- [ ] **Step 2: Tag the milestone**

```bash
cd /home/jiajli/apps/mlir-air
git tag -a llm-mapping-skills-v1 -m "$(cat <<'EOF'
LLM mapping skills v1: 11 skills, _llm_shared/ infra, smoke-tested on Llama-3.2-1B-Instruct

Implements the design spec at docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md.
Per-phase + cross-cutting + entry skills authored. Foundation lift complete.
Validated end-to-end via smoke test on Llama-3.2-1B-Instruct (trivial-delta arch).

Pilot deployments on TinyLlama-1.1B and SmolLM2-1.7B are explicit follow-on work.
EOF
)"
git log -1 --oneline
git tag -l "llm-mapping*"
```

- [ ] **Step 3: Push branch and tag**

```bash
cd /home/jiajli/apps/mlir-air
git push origin llm_mapping
git push origin llm-mapping-skills-v1
```

---

## Self-Review

**Spec coverage check:**
- §1 Motivation — context for the plan ✓
- §2 Scope — encoded in Task 14 (entry skill arch check) and Task 7 (bootstrap-model-config Step 2) ✓
- §3 Architecture — Tasks 2 (lift), 4–14 (skills creation), 16 (README) ✓
- §4 Skill inventory — Tasks 4–14 produce all 11 skills ✓
- §5 Phase verification gates — encoded in each per-phase skill's "Verification" section ✓
- §6 State contract (TODO.md) — Task 14 Step 5 produces it; per-phase skills update it ✓
- §7 Inner debug-loop — encoded in Task 8 Step 2 (validate-per-kernel-shapes) and Task 11 (optimize-prefill-perf) ✓
- §8 Knowledge base + skill template — every skill task uses the template; references cite llama3 docs ✓
- §9 Pilot models — Task 15 covers smoke test (Llama-3.2-1B-Instruct); TinyLlama and SmolLM2 explicitly deferred per "Out of scope" header ✓
- §10 Non-goals — respected (no perf-parity gate, no autonomous execution, etc.) ✓
- §11 Open questions — flagged in plan: HF weight remapping (Task 7 Step 3), lift breakage (Tasks 2–3), perf judgment calls (Tasks 11–12) ✓
- §12 Implementation order — followed (lift → recipes → per-phase → entry → validate → docs) ✓

**Placeholder scan:**
- All file paths are absolute and real
- All shell commands are runnable as written
- All skill content is provided in full (no "TBD")
- All cross-references between tasks/skills are concrete

**Type/naming consistency:**
- Skill names match between definitions (Tasks 4–14) and references (entry skill, per-phase skills, debug-loop)
- File paths match between Task 2 (lift target) and skill citations (`_llm_shared/kernel_builder/...`)
- Phase numbering consistent (0–6) throughout

No issues found.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-17-llm-mapping-skills.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because authoring 11 skills in parallel-ish saves context and lets each skill get a fresh, focused context.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review. Better if you want to closely supervise each step.

**Which approach?**
