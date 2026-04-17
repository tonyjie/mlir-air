# Lessons from llama32_1b_instruct deployment

## Lesson 1: File-rename step in `deploy-new-llm` is over-prescriptive
**What happened**: The skill's Step 4 says to rename `llama3_*.py` → `<model>_*.py`.
For an identical-arch model (instruct variant), this requires also rewriting
internal cross-file imports (`from llama3_weights import` → `from <model>_weights import`)
and Makefile references — significant friction for no semantic gain.

**Workaround used**: Skipped the rename. Used `MODEL=instruct` Makefile variable
(already supported by the existing `llama3_inference.py --model` arg) instead.

**Skill update needed**: Mark file rename as optional. Recommend it for
arch-divergent models where the new code paths will diverge meaningfully; skip
for variants that share the base's code structure. Revise `deploy-new-llm`
Step 4 accordingly.

## Lesson 2: `cp -r` copies build artifacts (GB-scale)
**What happened**: `cp -r llama3 <new_dir>` would copy `build_peano/` and
`prefill_kernel_cache/` (totaling several hundred MB to ~GB).

**Workaround used**: `cd llama3 && git ls-files | tar cf - --files-from=- |
(cd ../<new_dir> && tar xf -)` — copies only tracked files, ~1.5MB.

**Skill update needed**: Replace the `cp -r` step in `deploy-new-llm` with the
git-tracked-files-only approach.

## Lesson 3: Same-arch variants can symlink the build cache
**What happened**: Symlinking `build_peano -> ../llama3/build_peano` skipped the
4-minute kernel recompile because the shapes (and thus cache keys) are identical.

**Skill update needed**: When `bootstrap-model-config` detects that the new
model's resolved config matches an already-deployed model exactly, suggest
symlinking the build cache as a Phase 1 short-circuit.

## Lesson 4: Phase chain assumes every phase is fresh work
**What happened**: For an identical-arch deployment, Phases 1, 2, 4, 5 were
trivially "PASS by reference" — same kernels, same shapes, same multi-launch
ELFs. The `deploy-new-llm` skill currently has no notion of this fast path.

**Skill update needed**: Add a "config-fingerprint" check to `deploy-new-llm`
that detects exact match against an existing deployment and offers to
short-circuit phases that would just re-validate identical compute.
