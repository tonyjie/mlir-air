# Full Transformer Block Multi-Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce one LLAMA transformer block (15 steps) to a single multi-launch ELF — one `xrt.run()` per layer instead of ~8 kernel invocations. Extends the proven FFN multi-launch pattern to the entire block.

**Architecture:** Build each kernel module independently (with transforms), serialize to MLIR text, rename SSA values with unique prefixes, remap func-arg references, and assemble into one `func` with ~15 `air.launch` ops. All intermediates are shared DDR func arguments. ELF output format required.

**Tech Stack:** AIR MLIR, Python IR builders, text-based MLIR stitching (proven in `ffn_swiglu/run.py`), XRTBackend with ELF output.

---

## Critical Design Decision: Shape Compatibility Between Launches

The biggest challenge is that the current host pipeline does numpy **reshape/transpose** between certain kernel steps. In a multi-launch ELF, there's no host between launches. These reshapes must be handled.

### Shape transitions that need attention:

| From → To | Current host reshaping | Solution |
|-----------|----------------------|----------|
| Q GEMM output `(2048, 2048)` → RoPE Q input `(65536, 64)` | reshape + transpose | Use aliased func args with different memref types for same DDR buffer |
| K GEMM output `(2048, 512)` → RoPE K input `(16384, 64)` | reshape + transpose | Same aliasing approach |
| RoPE Q output `(65536, 64)` → FlashAttn Q input `(32, 2048, 64)` | reshape + transpose | Aliased args |
| RoPE K output `(16384, 64)` → FlashAttn K input `(8, 2048, 64)` | reshape + transpose | Aliased args |
| V GEMM output `(2048, 512)` → FlashAttn V input `(8, 2048, 64)` | reshape + transpose | Aliased args |
| FlashAttn output `(32, 2048, 64)` → O GEMM input `(2048, 2048)` | transpose + reshape | Aliased args |

**Key insight**: `reshape` is free — same data in DDR, different view. But `transpose` changes memory layout. The GEMM kernel writes data in row-major `(seq_len, emb_dim)` layout, while RoPE expects `(n_heads, seq_len, head_dim)` layout — these are NOT the same bytes in memory.

**Solution**: Each kernel must produce output in the layout its consumer expects. This means:
- GEMM Q produces `(seq_len, emb_dim)` which is the same bytes as `(seq_len, n_heads, head_dim)` — no transpose needed, just reshape
- But RoPE expects `(n_heads * seq_len, head_dim)` which IS a transposed view — the head dimension must be contiguous across rows

**This requires verifying** that the reshape from `(seq_len, n_heads, head_dim)` to `(n_heads * seq_len, head_dim)` is a pure reshape (contiguous) or requires transpose. If `q` is stored as `[row0_head0, row0_head1, ..., row0_head31, row1_head0, ...]` but RoPE needs `[head0_row0, head0_row1, ..., head0_row2047, head1_row0, ...]`, that's a transpose.

**Conclusion**: The transpose between GEMM output and RoPE input is a REAL data movement, not just a view. This means either:
1. Keep these as separate XRT invocations (with host transpose between)
2. Add a transpose launch between GEMM and RoPE
3. Modify RoPE kernel to accept the GEMM's output layout

---

## Phased Approach

### Phase 1: FFN Block (DONE ✅)
Steps 11-14: Gate GEMM + Up GEMM + SiLU×mul + Down GEMM → 4 launches, 1 ELF.
**Result**: 149ms → 52ms per layer.

### Phase 2: Attention Block (this plan)
Steps 2-4 + 8: Q/K/V/O GEMMs → 4 launches, 1 ELF.
No reshape needed — all GEMMs have compatible 2D memref types.

### Phase 3: Full Block (future)
All 15 steps → ~15 launches, 1 ELF.
Requires solving the transpose problem between GEMM→RoPE and RoPE→FlashAttn.

---

## Phase 2: Attention GEMMs Multi-Launch

### Architecture

```
func @attn_gemms(
    %input:  memref<2048x2048xbf16>,    # normed input (from RMSNorm)
    %wq:    memref<2048x2048xbf16>,     # Q weight
    %q_out: memref<2048x2048xbf16>,     # Q output
    %wk:    memref<2048x512xbf16>,      # K weight
    %k_out: memref<2048x512xbf16>,      # K output
    %wv:    memref<2048x512xbf16>,      # V weight
    %v_out: memref<2048x512xbf16>,      # V output
):
    air.launch 1: Q GEMM   input × wq → q_out    (same as gemm_qo)
    air.launch 2: K GEMM   input × wk → k_out    (same as gemm_kv)
    air.launch 3: V GEMM   input × wv → v_out    (same as gemm_kv)
    return
```

Three GEMM launches in one ELF. All share the same input buffer. No reshaping needed — GEMM outputs are standard 2D memrefs.

**Why not include O GEMM**: The O GEMM's input comes from FlashAttention output, which has a different shape `(32, 2048, 64)` → needs transpose to `(2048, 2048)`. Keeping O GEMM separate avoids this.

---

### Task 1: Build `attn_gemms_multi.py`

**Files:**
- Create: `programming_examples/llama3/attn_gemms_multi.py`

- [ ] **Step 1: Create the multi-launch attention GEMM builder**

Reuse the stitching approach from `ffn_swiglu/run.py`. Build 3 GEMM modules (Q, K, V) independently, rename, remap, assemble.

```python
def build_attn_gemms_module(seq_len, emb_dim, kv_dim):
    # Build Q GEMM: (seq_len, emb_dim) × (emb_dim, emb_dim) → (seq_len, emb_dim)
    q_mod = _build_gemm_module(seq_len, emb_dim, emb_dim, ...)
    # Build K GEMM: (seq_len, emb_dim) × (emb_dim, kv_dim) → (seq_len, kv_dim)
    k_mod = _build_gemm_module(seq_len, emb_dim, kv_dim, ...)
    # Build V GEMM: same shape as K
    v_mod = _build_gemm_module(seq_len, emb_dim, kv_dim, ...)

    # Stitch: 7 func args (input, wq, q_out, wk, k_out, wv, v_out)
    # Q: {0→0, 1→1, 2→2}
    # K: {0→0, 1→3, 2→4}
    # V: {0→0, 1→5, 2→6}
```

- [ ] **Step 2: Verify module builds and parses**

```bash
python3 attn_gemms_multi.py -p  # Print MLIR
```

- [ ] **Step 3: Compile and run correctness test**

```bash
python3 attn_gemms_multi.py --compile-and-run
```

- [ ] **Step 4: Profile and compare with 3 separate kernel calls**

- [ ] **Step 5: Commit**

---

### Task 2: Integrate into LLAMA pipeline

**Files:**
- Modify: `llama3/llama3_prefill.py`

- [ ] **Step 1: Replace steps 2-4 with single `attn_gemms_multi` call**

- [ ] **Step 2: Verify 1-layer correctness**

- [ ] **Step 3: Profile 16-layer and compare**

- [ ] **Step 4: Commit**

---

### Task 3: Phase 3 Investigation — Full Block Transpose Problem

**Files:**
- Create: `llama3/docs/plans/full_block_transpose_analysis.md`

- [ ] **Step 1: Verify reshape compatibility**

Check whether `(seq_len, n_heads, head_dim)` → `(n_heads * seq_len, head_dim)` is contiguous (pure reshape) or requires transpose.

```python
import numpy as np
q = np.random.rand(2048, 32, 64).astype(np.float32)
q_reshaped = q.transpose(1, 0, 2).reshape(32 * 2048, 64)
print(q_reshaped.flags['C_CONTIGUOUS'])  # If True, reshape is free
```

- [ ] **Step 2: If transpose needed, evaluate options**

a. Add a data transpose `air.launch` between GEMM and RoPE
b. Modify RoPE kernel to accept `(seq_len, n_heads, head_dim)` layout
c. Modify GEMM kernel to output in transposed layout

- [ ] **Step 3: Document findings and recommend approach for Phase 3**

- [ ] **Step 4: Commit**

---

## Expected Outcomes

### Phase 2 (Attention GEMMs Multi-Launch)
- Steps 2-4: 3 GEMM calls → 1 XRT invocation
- Savings: ~10-15ms/layer (eliminate 2 kernel dispatches + BO write/read overhead)
- Estimated per-layer: 160ms → ~150ms

### Phase 3 (Full Block, future)
- All 15 steps → 1 XRT invocation per layer
- Savings: ~30-40ms/layer (eliminate all inter-kernel host overhead)
- Estimated per-layer: ~120-130ms, approaching IRON's 152ms

---

## Buffer Count Summary

| Phase | Func args | Launches | Host invocations/layer |
|-------|-----------|----------|----------------------|
| Current | N/A | N/A | 8 kernel calls |
| Phase 1 (FFN, done) | 8 | 4 | 5 kernel calls |
| Phase 2 (Attn GEMMs) | 7 | 3 | 4 kernel calls |
| Phase 2+1 combined | 15 total | 7 | 3 kernel calls (RMSNorm, Attn+RoPE+FlashAttn, Add) |
| Phase 3 (Full Block) | ~25 | ~15 | 1 kernel call |
