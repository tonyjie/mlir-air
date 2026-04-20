---
name: validate-per-kernel-shapes
description: Phase 1 of LLM deployment — verify each unique kernel shape the model needs passes against CPU reference. Invoked by deploy-new-llm after Phase 0 gate passes.
---

## Purpose
Before integration, prove each individual kernel (RMSNorm, GEMM, GEMV, RoPE, FlashAttention, SwiGLU, eltwise add) works correctly on NPU2 at every shape the new model needs. This isolates per-shape failures from integration bugs.

## Knowledge base references
Read these BEFORE acting:
- `programming_examples/_llm_shared/docs/aie2p_hardware_limits.md` —
  **REQUIRED** for the BD-friendliness audit in Step 0.5 below. Documents
  the 4 AIE2P shim DMA / lowering limits we've discovered + the
  "BD-friendliness checklist" that prevents most Phase 2 / Phase 5 compile
  blowups.
- `programming_examples/_llm_shared/docs/kernels/gemm.md` — GEMM tile config strategy
- `programming_examples/_llm_shared/docs/kernels/gemv.md` — GEMV herd layouts (8×1 K=2048; extern rename for K=8192)
- `programming_examples/_llm_shared/docs/kernels/rmsnorm.md` — 8-tile broadcast strategy
- `programming_examples/_llm_shared/docs/kernels/rope.md` — RoPE LUT layout, half-split convention
- `programming_examples/_llm_shared/docs/kernels/flash_attention.md` — seq-first layout, causal masking
- `programming_examples/_llm_shared/docs/kernels/ffn_swiglu.md`, `silu_and_mul.md`
- `programming_examples/_llm_shared/docs/kernels/eltwise_add.md`

## Workflow

### Step 0: Variant audit — REQUIRED for FlashAttention (LESSON 3 from llama32_3b deployment, 2026-04-18)

When citing an existing lit test as proof that a kernel "supports head_dim=N",
you MUST identify which Python builder the lit test exercises AND verify
production code uses the same builder. **The two FA Python builders
(`attn_npu2.py` head-first vs `attn_npu2_seqfirst.py` seq-first) compile from
the same C++ kernel but have different Python IR / DMA patterns**, and lit
tests for one DO NOT validate the other.

For each kernel where lit-test coverage is being claimed:

1. Find the lit test command. Example:
   ```
   $ grep -E "RUN.*make.*run.*DK=|--dk\\s+\\d+" \
       programming_examples/<kernel_dir>/*.lit
   ```
2. Trace the Makefile target back to the Python script it invokes:
   ```
   $ grep -E "python3.*\\.py" programming_examples/<kernel_dir>/Makefile
   ```
3. Diff against the Python file production code imports. If they differ:
   the lit test does NOT validate the production code path; flag this in
   the classification table as **"NEEDS Phase 2 standalone validation"**
   instead of trusting the classification.

**Concrete example from llama32_3b**: I claimed `run_npu2_makefile_peano_llama3_8b.lit`
proved head_dim=128 worked. The lit test exercises `attn_npu2.py` (head-first).
But llama3_prefill imports `attn_npu2_seqfirst.py` (seq-first). The seq-first
`dk_chunks > 1` path is broken — never lit-tested upstream — and hangs at
runtime. Cost of this audit miss: ~half a deployment session debugging FA hangs.

If you find a coverage gap, add a TODO.md entry like:
> Phase 2 prerequisite: standalone validation of `<production_python_file>` at
> `(n_heads=N, n_kv_heads=K, lq=lk=L, dk=D)` — lit test only covers the
> `<other_python_file>` variant.

A reusable bisect harness (vary one axis at a time across n_heads, n_kv_heads,
lq, dk; report HANG/NaN/PASS per cell) is the right tool for this validation —
see `debug-fa-runtime-failure` for the template.

### Step 0.5: BD-friendliness audit (LESSONS 3-5 from qwen25_1_5b deployment, 2026-04-19)

Run the audit checklist from
`programming_examples/_llm_shared/docs/aie2p_hardware_limits.md`
"BD-friendliness audit checklist" section. It surfaces the 4 known
AIE2P shim/lowering walls BEFORE compile time:

```python
# Rule A — flag dims that aren't 1024-aligned (Phase 2 risk)
for name, dim in [("emb_dim", config.emb_dim),
                  ("hidden_dim", config.hidden_dim),
                  ("kv_dim", config.n_kv_heads * config.head_dim)]:
    if dim % 1024 != 0:
        # → likely needs padding (qwen25_pad.py reference) at seq_len ≥ 2048

# Rule B — flag if Down GEMV K > 8160 (Phase 5 risk)
if config.hidden_dim > 8160:
    # → use `down_k_split=N` in build_o_gemv_ffn_module
    #   where hidden_dim % N == 0 and N ≤ 255

# Rule C — flag if any GEMV's M needs care (Phase 5 risk)
for name, M in [("Q", n_heads*head_dim), ("K/V", n_kv_heads*head_dim),
                ("Gate/Up", hidden_dim), ("LM-head-partition", 16384)]:
    default_fires = (M // 64) * 2  # default tile_m=8, m_input=4, herd_m=8
    if default_fires > 127:
        # → tile_m=m_input bump to cut inner loop to 1

# Rule D — flag if Down GEMV L2 doesn't fit (Phase 5 risk)
if config.hidden_dim * 8 * 2 * 2 > 512 * 1024:
    # → smaller tile_m or herd_m
```

Append surfaced risks as TODO.md "Phase 2 prerequisites" or
"Phase 5 prerequisites" entries. **Each unsurfaced risk costs ~30 min
of debug detour at compile time.**

### Step 0.6: Tile-config safety check (LESSON 2 from qwen25_1_5b, 2026-04-19)

The shared GEMM/GEMV builders DO NOT assert these constraints — failures
produce SILENTLY CORRUPT outputs (e.g., qwen25 Phase 2 hit cosine 0.02
because seq_len=128 < tile_m*herd_m=512). Verify upfront:

For every GEMM/GEMV in your shape table, confirm:
- `N % (tile_n * herd_n) == 0` for each (Q, K, V, O, Gate, Up, Down)
- `M >= tile_m * herd_m` AND `M % (tile_m * herd_m) == 0`

If these don't hold for the default tile config, pick a different config
or pad. Specifically:
- Qwen2.5 K/V at N=256 needed `tile_n=64, herd_n=4` (256/256=1 ✓) instead
  of default `tile_n=128, herd_n=4` (256/512=0.5 ✗ silent corruption).

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
