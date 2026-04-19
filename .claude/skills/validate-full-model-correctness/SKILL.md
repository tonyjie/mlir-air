---
name: validate-full-model-correctness
description: Phase 3 of LLM deployment — wire all N layers and verify top-1 prediction matches CPU reference for canonical prompts; per-layer correlation > 0.95 throughout. Invoked after Phase 2 gate.
---

## Purpose
Scale single-block correctness to the full N-layer stack. Catches accumulated drift across layers, KV cache bugs, layer-specific weight loading errors, and BF16 truncation that compounds.

## Knowledge base references
- `programming_examples/llama3/llama3_inference.py:run_npu_prefill` — reference full-stack pipeline
- `programming_examples/llama3/docs/development_progress/progress.md` — LLAMA Phase 3 (Paris test, F32 accumulator fix)

## Workflow

### Step 1: Wire all N layers
Implement `run_full_prefill(input_ids, weights, config)` that loops `run_single_block(layer_idx, ...)` for `layer_idx in range(config.n_layers)`, then applies final RMSNorm + LM head.

### Step 2: Define canonical prompts
At minimum 6 prompts split into two classes (LESSON 2 from llama32_3b
deployment, 2026-04-18):

**Decisive prompts** (CPU softmax(logits)[top1] > 0.5 — top-1 is so dominant
that BF16 accumulation noise across N layers cannot reorder it):
- "1 + 1 =" (expected ' ' with p ≈ 0.74)
- "2 + 2 =" (expected ' ' with p ≈ 0.53)
- "Water freezes at" (expected ' ' / ' 0' with p ≈ 0.71)
- "The largest ocean is the" (expected ' Pacific' with p ≈ 0.82)

**Competitive prompts** (CPU top-1 prob ≤ 0.5 — multiple plausible
continuations within close probabilities; BF16 will reorder them):
- "The capital of France is" (CPU top-1 ' Paris' p ≈ 0.25, top-2 ' the' p ≈ 0.14)
- "The sky is" (CPU top-1 ' the' p ≈ 0.33, top-2 ' falling' p ≈ 0.07)

(For instruct/chat models, use chat-templated equivalents and re-measure CPU
top-1 prob to re-classify into decisive vs competitive.)

You will also need to capture the CPU top-5 token IDs for the competitive
prompts so the gate can check top-5 overlap. Compute and cache these once
during Phase 0 reference validation.

### Step 3: Run NPU vs reference for each prompt
Predict next token via:
- NPU full prefill → top-1 token
- CPU reference full prefill → top-1 token

### Step 4: If top-1 mismatch, run per-layer correlation analysis
For each layer 0..N-1:
- Compute `cosine_sim(npu_layer_out, ref_layer_out)`
- Identify the first layer where correlation drops below 0.95

That's the offending layer — bisect within the layer using Phase 2's per-kernel approach.

### Step 5: Document
Record per-layer correlation table in `<model>/docs/development_progress/phase3_full.md`.

## Verification (Phase 3 gate)

Phase 3 PASSES when ALL true:
- **Decisive prompts** (CPU top-1 p > 0.5): NPU top-1 matches CPU top-1 EXACTLY,
  for ALL decisive prompts in the canonical set
- **Competitive prompts** (CPU top-1 p ≤ 0.5): top-5 overlap — `CPU top-1 ∈ NPU
  top-5` AND `NPU top-1 ∈ CPU top-5`, for ALL competitive prompts
- No NaN anywhere in the stack

**Per-layer correlation > 0.95 is informational, NOT a gate** for models with
n_layers ≥ 24 OR head_dim ≥ 128. BF16 accumulation across 28 layers (e.g.
llama32_3b) drives per-layer cosine to ~0.88 by the last layer despite no
kernel bug — this is geometric drift, not a regression. The strongest
end-to-end correctness signal is the top-1/top-5 gate above. Use per-layer
cos > 0.95 as a *gate* only for shallower models (n_layers < 24, head_dim ≤ 64
— e.g. llama3-1B, smollm2-1.7B) where the noise budget supports it.

## Failure modes
- Decisive prompt top-1 mismatch → real correctness issue; bisect via per-layer
  diagnostic to find first layer where cosine drops below 0.95 → bisect within
  that layer per Phase 2's per-kernel approach (this is `verify=True` in
  `run_transformer_block`)
- Competitive prompt top-5 NON-overlap (NPU produces token NOT in CPU top-5)
  → real bug; bisect as above
- Per-layer corr drops at a single layer (cliff, not gradual) → likely a
  layer-indexed weight loading bug (check `bo_key=f"kernel_L{i}"`)
- Per-layer corr drifts gradually from layer 0 to a low number AND top-1 mismatch
  on a competitive prompt → almost certainly BF16 accumulation in deep stack
  (LESSON 2 pattern). Verify by adding 2-3 decisive prompts; if they all
  match top-1 the deployment is OK and you've just exposed the BF16 reorder
  on a borderline prompt.
- Per-layer corr stays > 0.99 BUT decisive prompt top-1 mismatch → likely LM
  head precision; invoke F32 accumulator pattern from `kernels/gemm.md`

## Update protocol

On Phase 3 PASS, this is the **end-to-end correctness milestone**. Update `progress.md` with full results table, mark `TODO.md` Phase 3, advance to perf phases.
