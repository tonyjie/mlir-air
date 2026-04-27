---
name: full-model-validation
description: Phase 3 of LLM deployment — wire all N layers and verify NPU output matches CPU reference numerically (per-layer cosine + final logits cosine + top-1/top-5 token match) at canonical prompts. Catches accumulated drift, KV cache bugs, layer-indexed weight loading errors. Invoked after Phase 2 gate.
---

## Purpose

Phase 2 verified one transformer block. Phase 3 scales to all N layers
and confirms the full prefill stays numerically aligned with the CPU
reference end-to-end. Catches: accumulated BF16 drift across deep
stacks, KV cache layout bugs, layer-indexed weight loading errors,
LM head precision drops.

## Phase 3 PASS criteria (HARD GATES)

Run NPU full prefill + `<model>_reference.py` full prefill on every
canonical prompt (Step 2). All must hold:

**Numerical correctness (primary gate, vs CPU reference)**

1. **Final logits cosine ≥ 0.95** at the prediction position. This is
   the end-to-end numerical signal — the LM head's output cosine after
   N layers of stacked computation.
2. **Per-layer cosine ≥ 0.85** at EVERY layer (`cosine_sim(npu_layer_i_out,
   ref_layer_i_out)` for i in [0, n_layers)). 0.85 is loose enough to
   admit legitimate BF16 drift in deep models (llama32_3b 28-layer
   stack hits ~0.88 at last layer with no kernel bug — LESSON 2), but
   tight enough to catch gross corruption.
3. **No sudden cliff** between consecutive layers:
   `|cos[i+1] - cos[i]| < 0.05`. Catches layer-indexed bugs (wrong
   `bo_key=f"kernel_L{i}"`, weight-load shifted by a layer) that
   would slip through (1) and (2) if the absolute value stays above
   threshold but the trend has a discontinuity.
4. **Record `max_abs` and `max_rel` error** alongside the cosines
   (informational, not gated — absolute thresholds depend on prompt
   and layer). Compare to the reference deployment's measured values
   for the same prompt — equal-or-better is no regression. Future
   deployments can use these recorded numbers as a regression baseline
   (e.g., "new deployment's per-layer max_abs ≤ 1.5× the value
   logged here at the same layer signals no perf-related correctness
   drop").

**Semantic confirmation (secondary, validates numerical gate maps to
correct behavior)**

5. **Decisive prompts** (CPU top-1 prob > 0.5): NPU top-1 strict ==
   CPU top-1, for ALL decisive prompts in the canonical set.
6. **Competitive prompts** (CPU top-1 prob ≤ 0.5): bidirectional
   top-5 overlap — `CPU top-1 ∈ NPU top-5` AND `NPU top-1 ∈ CPU top-5`.

**Hygiene**

7. No NaN anywhere in the stack.
8. Per-layer cos table + final logits cos + top-1/top-5 results for
   every canonical prompt documented in
   `<model>/docs/development_progress/phase3_full.md`.

## Knowledge base references

- `programming_examples/llama3/llama3_inference.py:run_npu_prefill` —
  reference full-stack pipeline (loops Phase 2's per-layer block)
- `programming_examples/llama3/llama3_inference.py` `--verify` mode —
  reference for per-layer K/V + final logits comparison harness
- `programming_examples/llama3/docs/development_progress/progress.md` —
  llama3 Phase 3 narrative (Paris test, F32 accumulator fix)
- `programming_examples/llama32_3b/docs/development_progress/LESSONS.md`
  Lesson 2 — decisive/competitive prompt classification, BF16 drift in
  deep stacks
- `programming_examples/_llm_shared/docs/kernel_registry/<model>.md`
  — Phase 1 verified kernels (Phase 3 confirms they compose at scale)

## Workflow

### Step 1: Wire all N layers

Implement `run_full_prefill(input_ids, weights, config)`:

- **Inheritance path**: loop `llama3_prefill.run_transformer_block(...)`
  for `layer_idx in range(config.n_layers)`, then apply final RMSNorm
  + LM head.
- **Kernel-first path**: loop `run_transformer_block_<model>(...)`
  (the per-model block runner Phase 2 produced) the same way.

Either way, this is just iterating the Phase 2 block runner N times
plus head — no new kernels.

### Step 2: Define canonical prompts

Use these prompts (or chat-templated equivalents for instruct models —
re-measure CPU top-1 prob to re-classify):

**Decisive** (CPU softmax(logits)[top1] > 0.5 — top-1 dominates so
clearly that BF16 noise across N layers cannot reorder it):

- `"1 + 1 ="` (expected ` 2`, p ≈ 0.74)
- `"2 + 2 ="` (expected ` 4`, p ≈ 0.53)
- `"Water freezes at"` (expected ` 0` / ` zero`, p ≈ 0.71)
- `"The largest ocean is the"` (expected ` Pacific`, p ≈ 0.82)

**Competitive** (CPU top-1 prob ≤ 0.5 — multiple plausible
continuations within close probability; BF16 will reorder them):

- `"The capital of France is"` (CPU top-1 ` Paris` p ≈ 0.25, top-2 ` the` p ≈ 0.14)
- `"The sky is"` (CPU top-1 ` the` p ≈ 0.33, top-2 ` falling` p ≈ 0.07)

Cache CPU top-5 token IDs for competitive prompts during Phase 0
reference validation so Phase 3 can check overlap directly.

### Step 3: Run NPU vs reference + collect numerical metrics

For each canonical prompt:

1. NPU full prefill → capture per-layer hidden states `npu_layer_out[i]`
   for i in [0, n_layers), final logits, and top-5 token IDs at the
   prediction position.
2. CPU reference full prefill → same quantities from
   `<model>_reference.py`.
3. Compute `cosine_sim(npu_layer_out[i], ref_layer_out[i])` for every
   `i`. Compute final logits cosine + top-1 / top-5 match.

Check against PASS criteria above.

### Step 4: Bisect on FAIL

If a numerical gate fails, the per-layer cosine table localizes
where:

- **Sudden cliff at layer i** (cos[i] >> cos[i+1]) → layer-indexed bug
  at i+1 (weight load shifted, wrong `bo_key`, wrong `wq` for that layer)
- **Gradual drift** but final < 0.95 → BF16 accumulator saturating;
  check whether production GEMM path uses F32 internal accumulate;
  consider whether decisive top-1 still matches (if YES, deployment
  is fine — drift is borderline; if NO, real bug)
- **Layer 0 already low** → integration error in single-block runner
  itself; revisit Phase 2

Within the offending layer, bisect kernel-by-kernel using Phase 2's
CPU-fallback technique (swap NPU kernel back to reference, find
boundary).

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Per-layer cos cliff at one layer | Layer-indexed weight load bug or wrong `bo_key=f"kernel_L{i}"` | Print weight shapes per layer; compare K/V cache layout at boundary |
| Per-layer cos drifts gradually < 0.85 by last layer | Real BF16 saturation, OR Down GEMM missing F32 accumulator | Run a decisive prompt — if top-1 matches, drift is geometric not bug; otherwise check `kernels/gemm.md` F32 accumulator pattern |
| Per-layer cos OK but final logits cos < 0.95 | LM head precision drop (BF16 truncation in vocab projection) | F32 accumulator pattern from `kernels/gemm.md` applied to LM head GEMM |
| Decisive prompt top-1 mismatch | Real correctness issue regardless of cos | Numerical gates probably also failed — Step 4 bisect |
| Competitive prompt top-5 NON-overlap (NPU top-1 NOT in CPU top-5 at all) | Real bug | Step 4 bisect |
| NaN in output | Uninitialized BO / reused stale buffer | Invoke `debug-bo-corruption` |
| Per-layer cos OK, top-1 OK, but multi-token generation diverges quickly | KV cache update bug at decode time (Phase 3 only tests prefill — drift compounds during decode) | Validate KV cache values at end of prefill match reference |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 3 PASS, this is the **end-to-end correctness milestone** —
NPU is now numerically faithful to the CPU reference at full N-layer
scale. Update:

- `<model>/docs/development_progress/phase3_full.md`: per-layer cos
  table + final logits cos + top-1/top-5 results per prompt
- `<model>/docs/development_progress/progress.md`: Phase 3 summary
- `<model>/TODO.md`: mark Phase 3, advance to perf phases

Phase 4 (prefill perf) and Phase 5 (decode perf) MUST preserve these
numerical gates after every optimization — perf cannot trade
correctness.
