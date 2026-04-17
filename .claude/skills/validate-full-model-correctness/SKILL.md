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
At minimum 3 prompts that have unambiguous next-token completions in the reference model:
- "The capital of France is" (expected: " Paris" or similar)
- "1 + 1 = " (expected: "2")
- "The sky is" (expected: " blue" or similar)

(For instruct/chat models, use chat-templated equivalents.)

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

Phase 3 PASSES when:
- Top-1 NPU prediction matches CPU reference for ≥3 of 3 canonical prompts
- Per-layer correlation > 0.95 for all layers (no drift)
- No NaN anywhere in the stack

## Failure modes
- Top-1 mismatch but per-layer corr stays > 0.95 → likely LM head precision; invoke F32 accumulator pattern from `kernels/gemm.md`
- Per-layer corr drops at a single layer → likely a layer-indexed weight loading bug (check `bo_key=f"kernel_L{i}"`)
- Drift starts at layer 0 and worsens → likely BF16 accumulator issue throughout; F32 accumulator fix per LLAMA Phase 3

## Update protocol

On Phase 3 PASS, this is the **end-to-end correctness milestone**. Update `progress.md` with full results table, mark `TODO.md` Phase 3, advance to perf phases.
