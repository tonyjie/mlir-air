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
- If not, wrap (see `_llm_shared/kernel_builder/stitching.py` for `wrap_herd_in_segment` helper if present, else add manually)

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
