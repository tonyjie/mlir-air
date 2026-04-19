# `_llm_shared/docs/` — Shared Infrastructure Documentation

Generic patterns, kernel design notes, and compiler-issue writeups that apply
across all LLM deployments on MLIR-AIR / NPU2. The skill chain
(`.claude/skills/`) cites these as canonical references.

These docs were originally written during the LLAMA-3.2-1B deployment (the
first end-to-end NPU2 LLM), which is why many use llama3 shapes as concrete
examples — but the patterns themselves are deployment-agnostic. Per-deployment
status logs, lessons, and progress reports stay under
`programming_examples/<model>/docs/`.

## Topic map

### `explain.md`
Compilation pipeline overview: how Linalg MLIR → AIR → AIE → ELF → xclbin
flows. Kernel directory map. RoPE half-split convention details. Read first
if you're new to MLIR-AIR.

### `perf_optimization.md`
The 18.67 s → 1.30 s prefill optimization journey on llama3-1B. Pattern
catalog: multi-launch merging, BO pre-loading, intermediate buffer reuse,
seq-first activation layout, CPU→NPU op promotion. Cited by Phase 4 + Phase 5
optimization skills as the canonical perf-pattern reference.

### `kernels/`
Per-kernel design notes — algorithm, tile config strategy, herd layout, BF16
precision considerations. One file per kernel:

| File | What it covers |
|---|---|
| `gemm.md` | GEMM tile config strategy, 8×4 herd, BF16-output rounding fix |
| `gemv.md` | Decode GEMV herd layouts, K=2048 vs K=8192 partitioning, extern kernel rename |
| `rmsnorm.md` | 8-tile broadcast strategy, weight-broadcast DMA |
| `rope.md` | RoPE LUT layout (concatenated cos/sin), half-split convention vs interleaved |
| `flash_attention.md` | Seq-first vs head-first FA, causal masking, dk_chunks > 1 path |
| `ffn_swiglu.md` | FFN block multi-launch (gate/up/down + SiLU+mul) |
| `silu_and_mul.md` | SiLU activation + elementwise multiply |
| `eltwise_add.md` | Residual add kernel optimization |

### `multi-launch/`
Multi-launch ELF stitching: how multiple `air.launch` ops fuse into one
`xrt.run()` invocation, reducing dispatch overhead.

| File | What it covers |
|---|---|
| `host_optimization.md` | `static_input_indices`, `intermediate_indices`, BO write/read overhead |
| `decode_merging.md` | Decode-specific merge patterns (extern kernel rename for K=8192) |
| `full_block.md` | What does NOT merge (FlashAttention) and why |
| `compiler_scaling.md` | Compile-time scaling with ELF size |

### `compiler_issues/`
Known MLIR-AIR compiler quirks and workarounds.

| File | What it covers |
|---|---|
| `herd_load_bug.md` | `airrt.herd_load` failing silently when herd is not wrapped in `air.segment` |
| `multi_launch_blockers.md` | Exhaustive list of constraints that reject a multi-launch merge |
| `multi_launch_root_cause.md` | Root cause analyses for the merge blockers |
| `weight_broadcast_dma.md` | DMA stride=0 limitation for sub-32b weight broadcast |

## Where else to look

- **Live source code** (helper modules, kernel builders): `../kernel_builder/` + `../phase_helpers/`
- **The skill chain itself**: `.claude/skills/` (12 skills + recipes)
- **Per-deployment status & lessons**: `programming_examples/<model>/docs/development_progress/`
- **llama3 history not directly relevant to skills**: `../llama3/docs/{usage,profile,issues}.md` and `../llama3/docs/development_progress/{plan,progress,decode_archive/}`. (Llama3-specific narrative; the skill chain doesn't cite these.)

## Maintenance

When you add a new generic pattern doc:
1. Put it under the appropriate subdirectory (or create one if it's a new topic).
2. Add a row to the topic map above.
3. If a skill should cite it, add a "Knowledge base references" entry in that skill's `SKILL.md`.

When you add a per-deployment status doc, write it under `<model>/docs/development_progress/` instead — even if it documents a generic pattern, the writeup is initially deployment-specific. Promote to `_llm_shared/docs/` only if a 2nd deployment validates the same pattern (the "smollm2 finalize" criterion).
