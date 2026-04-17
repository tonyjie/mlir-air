# llama32_1b_instruct — smoke-test artifact

This directory documents a successful smoke test of the `deploy-new-llm` skill chain
on `meta-llama/Llama-3.2-1B-Instruct` (2026-04-17).

## What's here
- `TODO.md` — deployment phase tracker (all 7 phases PASSED)
- `docs/development_progress/progress.md` — phase-by-phase results
- `docs/development_progress/LESSONS.md` — skill-refinement lessons learned
- `docs/development_progress/debug_log.md` — (empty; no debug recipes fired)
- `Makefile` — would-be entry point if this were a divergent-arch deployment

## What's NOT here (and why)

The Python source files (`llama3_*.py`, `multi_launch_builder/`) and the docs that
live in `programming_examples/llama3/` are **not duplicated** here. The instruct
variant has identical architecture to `Llama-3.2-1B` base, so all code paths and
kernel binaries are reused via:

```bash
cd ../llama3
make run MODEL=instruct        # the actual command to run instruct on NPU
```

The `MODEL=instruct` Makefile variable in `llama3/` already supports this; that's
how the smoke test validated end-to-end behavior.

## When *would* a new-model dir be self-contained?

For an **arch-divergent** new model (e.g., TinyLlama-1.1B with different layer
count, GQA ratio, RoPE base), the deploy-new-llm skill would scaffold a full
copy with adapted Python files. This dir's lean scaffold reflects the
arch-identical fast path — see `LESSONS.md` for the skill refinement that
should formalize this case.
