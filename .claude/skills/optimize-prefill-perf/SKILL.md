---
name: optimize-prefill-perf
description: Phase 4 of LLM deployment — apply the 5 known prefill optimization patterns from LLAMA Phase 4 (multi-launch merging, BO pre-loading, intermediate buffer reuse, seq-first layout, CPU→NPU op promotion). Invoked after Phase 3 correctness gate.
---

## Purpose
Apply the prefill optimization patterns that took LLAMA from 18.67s → 1.30s. Each pattern is mechanically applicable; the skill attempts each and records the gain or skip-reason. Judgment-heavy choices defer to human.

## Knowledge base references
- `programming_examples/_llm_shared/docs/perf_optimization.md` — full optimization journey
- `programming_examples/_llm_shared/docs/multi-launch/host_optimization.md` — `static_input_indices`, `intermediate_indices`
- `programming_examples/_llm_shared/docs/multi-launch/decode_merging.md` — extern kernel rename pattern (used here for shape-collision avoidance)

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

**At head_dim ≥ 128, NPU FlashAttention requires special handling** (LESSON 3
from llama32_3b deployment, 2026-04-18). Two pitfalls:

1. **`compile_attn_npu2*` flag conventions are per-tile, NOT per-launch.**
   The Makefile uses `-Dlqp=$(LQP_TILE)` where `LQP_TILE = LQP / NUM_Q_TILES`,
   `-Ddk=$(LKP)` (NOT `DK`), and `-Ddk_full=$(DK)`. Passing per-launch sizes
   (e.g., `lqp=256, dk=128`) to the .o produces a kernel doing the wrong
   per-tile arithmetic — output is all-NaN at runtime. The fixed
   `compile_attn_npu2_split(lqp, lkp, dk, dv, num_q_tiles=4)` API in
   `_llm_shared/kernel_builder/external_kernels.py` derives `lqp_tile` and
   emits the correct flags.

2. **The seq-first FA `dk_chunks > 1` path hangs upstream.** Production llama3
   imports `attn_npu2_seqfirst.py`, which has untested code paths for
   `dk_chunks > 1`. At head_dim=128 with `lkp=64` (the only L1-feasible config),
   `dk_chunks=2` and the kernel hangs with `ERT_CMD_STATE_TIMEOUT`. **Workaround
   (Option C, proven on llama32_3b)**: switch to head-first `attn_npu2.py` +
   wrap with host transposes via a `_run_cached("flash_attn", ...)` monkey-patch.
   **Reusable implementation:**
   `programming_examples/_llm_shared/phase_helpers/headfirst_fa.py` exposes
   `install_headfirst_fa_wrapper()` + `compile_headfirst_fa_kernel(...)`;
   `_llm_shared/phase_helpers/orchestration.compile_block_kernels` already
   auto-routes head_dim ≥ 128 through it. Cost: a few ms/layer host transpose;
   gain: 4.2× speedup vs CPU-attn (llama32_3b warm prefill 13.6 s → 3.2 s).

If FA hangs or produces NaN at head_dim ≥ 128, invoke `debug-fa-runtime-failure`
recipe BEFORE assuming the kernel is broken — it bisects (n_heads, n_kv, lq=lk,
dk) one axis at a time and discriminates between .o flag bugs (NaN), seq-first
dk_chunks bugs (HANG), and the L1-budget situation.

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
- NPU FA hangs / produces NaN at head_dim ≥ 128 → `debug-fa-runtime-failure`
- Correctness regresses → revert the pattern, document reason

## Update protocol

Append to `progress.md` per-pattern table with measured gains. Update `TODO.md` Phase 4.
