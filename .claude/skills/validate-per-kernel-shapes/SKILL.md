---
name: validate-per-kernel-shapes
description: Phase 1 of LLM deployment — verify each unique kernel shape the model needs passes against CPU reference. Invoked by deploy-new-llm after Phase 0 gate passes.
---

## Purpose
Before integration, prove each individual kernel (RMSNorm, GEMM, GEMV, RoPE, FlashAttention, SwiGLU, eltwise add) works correctly on NPU2 at every shape the new model needs. This isolates per-shape failures from integration bugs.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/llama3/docs/development_progress/kernels/gemm.md` — GEMM tile config strategy
- `programming_examples/llama3/docs/development_progress/kernels/gemv.md` — GEMV herd layouts (8×1 K=2048; extern rename for K=8192)
- `programming_examples/llama3/docs/development_progress/kernels/rmsnorm.md` — 8-tile broadcast strategy
- `programming_examples/llama3/docs/development_progress/kernels/rope.md` — RoPE LUT layout, half-split convention
- `programming_examples/llama3/docs/development_progress/kernels/flash_attention.md` — seq-first layout, causal masking
- `programming_examples/llama3/docs/development_progress/kernels/ffn_swiglu.md`, `silu_and_mul.md`
- `programming_examples/llama3/docs/development_progress/kernels/eltwise_add.md`

## Workflow

### Step 1: Enumerate the unique shapes from `<model>_weights.py` Config
Compute the set of (kernel_type, shape_tuple) pairs the model needs:

```python
shapes = []
# RMSNorm: per-block input
shapes.append(("rmsnorm", (config.emb_dim,)))
# GEMM: prefill (M=seq_len, N, K=emb_dim) for Q/K/V/O/Gate/Up/Down
for (m, n, k) in [
    (config.seq_len, config.n_heads * config.head_dim, config.emb_dim),  # Q
    (config.seq_len, config.n_kv_heads * config.head_dim, config.emb_dim),  # K
    (config.seq_len, config.n_kv_heads * config.head_dim, config.emb_dim),  # V
    (config.seq_len, config.emb_dim, config.n_heads * config.head_dim),  # O
    (config.seq_len, config.hidden_dim, config.emb_dim),  # Gate
    (config.seq_len, config.hidden_dim, config.emb_dim),  # Up
    (config.seq_len, config.emb_dim, config.hidden_dim),  # Down
    (config.seq_len, config.vocab_size, config.emb_dim),  # LM head
]:
    shapes.append(("gemm", (m, n, k)))
# GEMV: decode (M=1) versions of the above
# RoPE: head_dim
# FlashAttention: (seq_len, n_heads, head_dim, n_kv_heads)
# SwiGLU/SiLU+Mul: hidden_dim
# Eltwise add: emb_dim
```

Deduplicate. Most LLAMA-derived models produce 12–16 unique shape tuples.

### Step 2: Loop over the inner debug-loop pattern (per spec §7)

For each `(kernel_type, shape)`:

```
1. Build the kernel module via _llm_shared/kernel_builder/<builder>
2. Use XRTRunner.run_test(...) with random inputs and CPU reference outputs
3. If passed → record pass, advance
4. If failed:
   a. Match error against debug-bo-corruption trigger pattern → if match, invoke that recipe
   b. Else invoke superpowers:systematic-debugging
   c. If still failed → escalate
```

Bound: 1 retry per recipe per shape; 1 systematic-debugging attempt per shape.

### Step 3: Produce a pass/fail table
Write to `<model>/docs/development_progress/phase1_kernel_shapes.md`:

```
| Kernel | Shape | Status | Recovered via |
|--------|-------|--------|---------------|
| RMSNorm | (2048,) | PASS | — |
| GEMM | (2048, 2048, 2048) | PASS | — |
| GEMM | (2048, 5632, 2048) | PASS_RECOVERED | debug-bo-corruption (1) |
| ...
```

## Verification (Phase 1 gate)

Phase 1 PASSES when ALL shapes show `PASS` or `PASS_RECOVERED` in the table. Any `FAIL` blocks the gate.

## Failure modes
- Shape `FAIL` after recipe + systematic-debugging → escalate to human via TODO.md "Active blockers"
- Compilation hang (>10 min) → likely a compiler-scaling issue; cap and document

## Update protocol

On Phase 1 PASS:
- Update `<model>/TODO.md`: mark Phase 1, append "(N/N PASSED)"
- Append phase1_kernel_shapes.md content (or summary) to `<model>/docs/development_progress/progress.md`
