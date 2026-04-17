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
