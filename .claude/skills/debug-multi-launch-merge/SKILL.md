---
name: debug-multi-launch-merge
description: Use when attempting to merge multiple kernel launches into a single multi-launch ELF and the compile fails (BD exhaustion, channel routing, herd shape conflict, IR validation error). Recipe for diagnosing and recovering from kernel-fusion failures.
---

## Purpose
When stitching kernels together via `_llm_shared/kernel_builder/stitching.py` to produce a multi-launch ELF, the AIE compiler often rejects the merged module for hardware-resource reasons (BD count, channel count, herd shape, routing). This skill diagnoses which constraint was hit and applies the appropriate workaround.

## Knowledge base references
Read these BEFORE acting if they apply:
- `programming_examples/_llm_shared/docs/compiler_issues/multi_launch_blockers.md` — exhaustive list of merge constraints
- `programming_examples/_llm_shared/docs/compiler_issues/multi_launch_root_cause.md` — root cause analyses
- `programming_examples/_llm_shared/docs/multi-launch/full_block.md` — why attention cannot merge further
- `programming_examples/_llm_shared/docs/multi-launch/compiler_scaling.md` — compile time scaling with ELF size
- `programming_examples/_llm_shared/docs/compiler_issues/weight_broadcast_dma.md` — DMA stride limitation for BF16 broadcast (also: memory `project_bf16_dma_stride`)

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
