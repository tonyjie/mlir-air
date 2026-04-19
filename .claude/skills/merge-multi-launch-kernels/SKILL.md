---
name: merge-multi-launch-kernels
description: Use when wanting to fuse multiple separate kernel launches into a single multi-launch ELF (single XRT invocation). The procedural recipe for the actual merge operation. Invoked by optimize-prefill-perf and optimize-decode-perf phase skills.
---

## Purpose
The procedure for merging N separate kernel launches into one multi-launch ELF using `_llm_shared/kernel_builder/stitching.py`. Reduces XRT dispatch overhead (each invocation costs ~50–200 µs on NPU2). For LLAMA: 10 launches/layer → 3 launches/layer was the major prefill perf win.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/_llm_shared/docs/multi-launch/host_optimization.md` — host-side BO write/read overhead, why merging helps
- `programming_examples/_llm_shared/docs/multi-launch/decode_merging.md` — decode-specific merge patterns (extern kernel rename for K=8192)
- `programming_examples/_llm_shared/docs/multi-launch/full_block.md` — what does NOT merge (attention)
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
