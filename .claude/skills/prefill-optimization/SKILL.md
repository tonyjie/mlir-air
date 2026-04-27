---
name: prefill-optimization
description: Phase 4 of LLM deployment — apply known prefill optimization patterns to a Phase-3-correct pipeline (multi-launch merge, BO pre-loading, intermediate buffer reuse, seq-first layout, CPU→NPU op promotion). Each step preserves correctness by re-running the Phase 3 numerical gate. Invoked after Phase 3 PASS.
---

## Purpose

Phase 1-3 produced a functionally correct NPU pipeline (kernel-by-kernel
in Phase 1, layer-by-layer in Phase 2-3, all numerically aligned with
the CPU oracle). Phase 4 keeps that correctness while applying
optimization patterns to **reduce prefill latency**. Every pattern is
an experiment: apply, re-measure, re-run Phase 3 gate; revert if
correctness regresses.

The reference deployment llama3-1B took prefill from 18.67 s → 1.30 s
(14×) by composing the 4 categories below. Most new deployments hit
within 2× of this trajectory once they apply Pattern A (the dominant
win).

## Phase 4 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied pattern, the Phase 3
   PASS criteria still hold (per-layer cosine ≥ 0.85 + no cliff;
   final logits cosine ≥ 0.95; top-1/top-5 match). Re-run Phase 3 gate
   between patterns. If correctness regresses, **revert the pattern**
   and document why it doesn't apply to this model.
2. **Prefill kernel time strictly < Phase 3 baseline**, measured at
   the same canonical prompt + seq_len with 5-warmup + 20-iter
   profile. Recording wall time too (kernel + host overhead) is good
   but the gate is on kernel time (host overhead optimization is a
   separate concern).
3. **Per-pattern outcome documented** in
   `<model>/docs/development_progress/phase4_prefill.md`: for each of
   the 4 patterns, record `applied / skipped / reverted`, the latency
   delta, and a one-line reason.

The "≥ N patterns applied" check is NOT a gate — some models
legitimately need only 1-2 patterns (e.g., the model is already
seq-first by construction → Pattern C N/A; no CPU fallback exists →
Pattern D N/A). The gate is the outcome (perf improved + correctness
preserved), not the process count.

## Knowledge base references

PRIMARY:

- `programming_examples/_llm_shared/docs/perf_optimization.md` — full
  llama3-1B optimization journey (18.67 s → 1.30 s prefill); the
  reference for what "good" looks like
- `programming_examples/<model>/docs/development_progress/phase3_full.md`
  — Phase 3 baseline timings + cosine numbers (the "before" state to
  measure against and preserve)
- `programming_examples/_llm_shared/docs/multi-launch/host_optimization.md`
  — `static_input_indices`, `intermediate_indices` mechanics

INHERITANCE PROTOTYPES (Pattern A reuse path):

- `programming_examples/llama3/multi_launch_builder/rms_gemms_rope_multi.py`
  — fused 6-launch ELF for RMSNorm + Q/K/V GEMM + RoPE Q/K
- `programming_examples/llama3/multi_launch_builder/o_ffn_multi.py`
  — fused 8-launch ELF for O + add + RMSNorm + Gate/Up + SwiGLU + Down + add

KERNEL-FIRST PROTOTYPE (Pattern A new-build path):

- `programming_examples/qwen3_0_6b/multi_launch/` — model-specific
  fused ELFs when the kernel sequence diverges from llama (Q/K Norm,
  reordering, etc.)

## Workflow

### Step 1: Measure Phase 3 baseline

Before applying any pattern, capture the baseline prefill kernel time:

```bash
cd programming_examples/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 <model>_prefill.py --profile --warmup 5 --iterations 20
```

Record: kernel time (ms) + wall time + per-layer cosine numbers (from
Phase 3). These are the numbers every pattern below must match or beat.

### Step 2: Apply optimization patterns

Apply A → B → C → D. Between each, re-run the Phase 3 gate
(`--verify-vs-cpu`) and re-measure profile. Skip a pattern if the
trigger condition isn't met (skip reasons go in the log).

#### Pattern A — Multi-launch merge (the dominant win)

Stitch each leaf kernel's `air.launch` into a single `func.func` so
the host issues ONE `xrt.run()` per group instead of one per kernel.
Llama3 prefill went from 16-XRT-call/layer to 3-XRT-call/layer here.

Two paths (decision mirrors Phase 2's inheritance vs kernel-first
integration choice):

| Path | When | What to do |
|---|---|---|
| **Reuse existing fused ELF** | Per-layer kernel sequence matches llama bit-for-bit (RMSNorm → Q/K/V → RoPE → FA → O → add → RMSNorm → Gate/Up → SwiGLU → Down → add) | Import `llama3/multi_launch_builder/{rms_gemms_rope_multi, o_ffn_multi}` directly; pass new shape parameters. Zero ELF-building work. smollm2, llama32_3b, qwen25 all do this. |
| **Build new fused ELF** | Model has new ops, op reordering, or fused-launch boundary changes (Phase 2 chose kernel-first integration) | Write new builders in `<model>/multi_launch/` (qwen3-0.6B prototype). Invoke `merge-multi-launch-kernels` skill for the procedural recipe; on compile failure invoke `debug-multi-launch-merge`. |

Either way, expected gain: 2–4× XRT-call reduction → corresponding
wall-clock improvement.

#### Pattern B — Eliminate redundant host↔NPU data movement

**B1. Per-layer BO pre-loading** — weights are written ONCE during
setup, reused on every layer call:

- Allocate per-layer BOs using `bo_key=f"kernel_L{layer_idx}"`
- Write weights in `prepare_runtime()` (not on every prefill call)
- Pass `static_input_indices=[<weight_indices>]` on every
  `cache.load_and_run()` to skip re-write

**B2. Intermediate buffer reuse** — for buffers the kernel fully
overwrites (its outputs and scratch):

- Pass `intermediate_indices=[<output_indices>]` on
  `cache.load_and_run()` to skip the initial host write

Llama3 observed multi-second prefill savings from B1 alone.

#### Pattern C — Seq-first activation layout

If the kernels in your block accept seq-first `(seq, n_heads*head_dim)`
natively, eliminate any host-side transposes between RoPE / FlashAttention.

- RoPE: accept seq-first input
- FlashAttention: accept seq-first Q, K, V (this is `attn_npu2_seqfirst.py`
  for head_dim ≤ 64; for head_dim ≥ 128, the head-first wrapper at
  `_llm_shared/phase_helpers/headfirst_fa.py` does the host transpose
  precisely so the rest of the pipeline stays seq-first)

Skip Pattern C if the model already runs seq-first end-to-end (most
inheritance deployments do).

Expected gain: eliminate 1–4 host transposes per layer.

#### Pattern D — CPU→NPU op promotion

If anything in the prefill pipeline still falls back to CPU (e.g.,
residual add wrapped in `np.add`, intermediate small-shape RMSNorm,
host-side `softmax`), move it to NPU using the standalone harness
that Phase 1 already validated.

Triggering condition: profile shows host overhead per layer ≥ kernel
time per layer. If kernel time dominates, skip.

> **head_dim ≥ 128 FA caveat**: NPU FlashAttention has known compile-
> flag and runtime quirks at head_dim ≥ 128 (LESSON 3). The reusable
> head-first wrapper auto-routes via
> `_llm_shared/phase_helpers/orchestration.compile_block_kernels`. If
> FA hangs or produces NaN, invoke **`debug-fa-runtime-failure`** —
> it bisects the three known root causes. Do NOT debug FA inline in
> this skill.

### Step 3: Re-run Phase 3 gate after each pattern

After every applied (or attempted) pattern:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock \
  python3 <model>_inference.py --verify --prompt "The capital of France is"
```

Confirm Phase 3 gate still passes (per-layer cos, final logits cos,
top-1 match). If regressed, revert the pattern and document why.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict) | Rule A non-1024-aligned dim OR wrong stitching boundary | Invoke `debug-multi-launch-merge` — it discriminates the 4 known compile blockers |
| Output corruption after BO pre-loading (correct first call, NaN/garbage on subsequent calls) | Per-layer BO key collision OR `static_input_indices` set wrong | Invoke `debug-bo-corruption` |
| FA hang (`ERT_CMD_STATE_TIMEOUT`) at head_dim ≥ 128 | Seq-first `dk_chunks > 1` path bug | Invoke `debug-fa-runtime-failure`; head-first wrapper is the workaround (see Pattern D caveat) |
| FA all-NaN at runtime | Compile-flag mismatch on `attn_npu2.cc` macros (LESSON 3 — `-Dlqp` must be per-tile, not per-launch) | Invoke `debug-fa-runtime-failure`; `compile_attn_npu2_split` derives correct flags |
| Cosine drops after Pattern X | Pattern X has a layout/type assumption your model violates | Revert Pattern X; check whether the assumption (e.g., seq-first only, all weights pre-transposed) holds |
| Latency unchanged after Pattern A | Multi-launch ELF compiled but XRT call count didn't drop | Check `xrt-smi top` for actual call count; verify the new fused ELF is what `_run_cached` actually invokes (not falling back to per-kernel path) |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 4 PASS:

- `<model>/docs/development_progress/phase4_prefill.md`: per-pattern
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 4, append final prefill kernel time +
  speedup vs Phase 3 baseline
- If a new fused ELF was built (kernel-first path of Pattern A),
  surface to Phase 6 for potential promotion to `_llm_shared/` if a
  second deployment validates the same pattern
