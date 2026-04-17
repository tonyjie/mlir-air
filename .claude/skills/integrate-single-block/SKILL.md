---
name: integrate-single-block
description: Phase 2 of LLM deployment — assemble one transformer block on NPU and verify cosine similarity > 0.99 against CPU reference. Invoked by deploy-new-llm after Phase 1 gate.
---

## Purpose
Once individual kernels work, integrate them into a single transformer block. This catches integration bugs (wrong tensor layouts between kernels, missing transposes, intermediate type mismatches) before scaling to N layers.

## Knowledge base references
- `programming_examples/llama3/llama3_prefill.py:run_transformer_block` — reference single-block pipeline
- `programming_examples/llama3/docs/development_progress/progress.md` — LLAMA Phase 2 log (CPU fallback strategy)
- `programming_examples/llama3/docs/explain.md` — kernel directory map

## Workflow

### Step 1: Wire one block with all-NPU kernels (no CPU fallback initially)
In `<model>/<model>_prefill.py`, implement `run_single_block(layer_idx, hidden, weights, ...)` that:
1. RMSNorm hidden → norm_out
2. Q/K/V GEMM on norm_out using layer weights
3. RoPE on Q and K
4. FlashAttention(Q, K, V) → attn_out
5. O GEMM on attn_out
6. Residual add: hidden + o_out → res1
7. RMSNorm res1 → norm2
8. Gate/Up GEMM on norm2
9. SiLU(Gate) * Up → swiglu_out
10. Down GEMM on swiglu_out
11. Residual add: res1 + down_out → block_out

### Step 2: Define correlation metric
```python
def cosine_sim(a, b):
    a_flat, b_flat = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def mae(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
```

### Step 3: Run NPU vs reference for layer 0
Pick a fixed input (e.g., embeddings of "The capital of France is"). Run both:
- NPU single block at layer 0 → npu_out
- `<model>_reference.py` single block at layer 0 → ref_out

Compute `cosine_sim(npu_out, ref_out)` and `mae(npu_out, ref_out)`.

### Step 4: If correlation < 0.99, bisect with CPU fallback
Replace each kernel one at a time with its CPU reference equivalent (using ops from `<model>_reference.py`). Find the kernel after which correlation drops below 0.99. That kernel is the offender — invoke `superpowers:systematic-debugging` on it.

### Step 5: Document
Record correlation table per kernel-bisect step in `<model>/docs/development_progress/phase2_block.md`.

## Verification (Phase 2 gate)

Phase 2 PASSES when:
- `cosine_sim(npu_block_out, ref_block_out) > 0.99` for layer 0
- `mae(npu_block_out, ref_block_out) < 1e-2`
- No NaN in NPU output

## Failure modes
- Correlation drops at Q/K/V GEMM → likely weight loading or tensor layout (check seq-first vs heads-first)
- Correlation drops at FlashAttention → likely causal masking missing or wrong (see `kernels/flash_attention.md`)
- Correlation drops at Down GEMM → BF16 truncation; check F32 accumulator (LLAMA Phase 3 fix per `progress.md`)
- NaN in output → invoke `debug-bo-corruption`

## Update protocol

On Phase 2 PASS:
- Append phase2 results to `progress.md`
- Update `TODO.md` Phase 2 checkbox
