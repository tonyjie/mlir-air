# Easy Merges Multi-Launch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-layer XRT invocations from 9 to 5 by merging non-transpose-dependent steps. No new kernel types needed — just stitching existing working launches together.

**Architecture:** Extend the proven text-based MLIR stitching to merge adjacent kernel groups that don't require host-side transpose between them.

**Tech Stack:** Same as FFN/attn_gemms multi-launch — text-based MLIR stitching, ELF format.

---

## Current State (9 XRT invocations per layer)

```
Invocation 1: rmsnorm              ~6ms
Invocation 2: attn_gemms (Q+K+V)   ~8ms
              --- HOST TRANSPOSE (Q,K for RoPE) ---
Invocation 3: rope_q               ~10ms
Invocation 4: rope_k               ~3ms
              --- HOST TRANSPOSE (Q,K,V for FlashAttn) ---
Invocation 5: flash_attn           ~22ms
              --- HOST TRANSPOSE (attn_out for O GEMM) ---
Invocation 6: gemm_qo (O proj)    ~8ms
Invocation 7: add (residual 1)     ~5ms
Invocation 8: rmsnorm              ~6ms
Invocation 9: ffn_multi + add      already 4+1? No, ffn_multi is separate from add
              Actually: ffn_multi (~50ms) + add (~5ms) = 2 invocations
```

Wait — let me recount from the actual code:

```
1. rmsnorm (pre-attn)           ~6ms   xclbin
2. attn_gemms (Q+K+V)           ~8ms   ELF (3 launches)
   --- HOST TRANSPOSE ---
3. rope_q                       ~10ms  xclbin
4. rope_k                       ~3ms   xclbin
   --- HOST TRANSPOSE ---
5. flash_attn                   ~22ms  ELF
   --- HOST TRANSPOSE ---
6. gemm_qo (O proj)            ~8ms   xclbin
7. add (residual 1)             ~5ms   xclbin
8. rmsnorm (pre-FFN)            ~6ms   xclbin
9. ffn_multi (Gate+Up+SiLU+Down) ~50ms ELF
10. add (residual 2)            ~5ms   xclbin
```

**10 XRT invocations** (I miscounted earlier). Total kernel: ~123ms. Host overhead: ~17ms.

---

## Merge Opportunities (No Transpose Required)

### Merge A: RoPE Q + RoPE K → 1 ELF

Both RoPE kernels accept flat 2D memrefs `(N, head_dim)`. No reshape between them. They just process different data sizes.

```
func @rope_qk(
    %q_in:   memref<65536x64xbf16>,   # Q RoPE input
    %lut_q:  memref<65536x64xbf16>,   # Q LUT (tiled)
    %q_out:  memref<65536x64xbf16>,   # Q RoPE output
    %k_in:   memref<16384x64xbf16>,   # K RoPE input
    %lut_k:  memref<16384x64xbf16>,   # K LUT (tiled)
    %k_out:  memref<16384x64xbf16>,   # K RoPE output
):
    air.launch: RoPE Q
    air.launch: RoPE K
    return
```

6 func args. Saves 1 XRT invocation.

**Estimated savings**: ~2-3ms (eliminate 1 dispatch + BO write overhead)

### Merge B: O GEMM + Residual Add → 1 ELF

The O projection writes `(seq_len, emb_dim)` and the residual add reads `(seq_len * emb_dim)` — same data, just flat vs 2D view. Can use `memref.collapse_shape` (like SwiGLU 2D→1D).

```
func @o_proj_add(
    %attn_out:  memref<2048x2048xbf16>,   # O GEMM input (from FlashAttn)
    %wo:        memref<2048x2048xbf16>,   # O weight
    %proj_out:  memref<2048x2048xbf16>,   # O GEMM output (intermediate)
    %x_residual: memref<4194304xbf16>,    # residual input (flat)
    %res_out:   memref<4194304xbf16>,     # residual output (flat)
):
    air.launch: O GEMM
    air.launch: Residual Add (reads proj_out as flat + x_residual)
    return
```

But wait — the eltwise_add kernel reads flat 1D, while GEMM outputs 2D. Need `collapse_shape` (same pattern as SwiGLU 2D→1D). And the residual add needs TWO inputs: the GEMM output and the original residual (`x_bf16`). This makes the arg mapping more complex.

Actually, the residual add takes: `a = x_bf16.flatten()` and `b = proj.flatten()`. Both are `(seq_len * emb_dim)` flat. The `proj` comes from O GEMM. If we use 2D/1D aliased args (like FFN), it works.

**Estimated savings**: ~3-4ms

### Merge C: RMSNorm (pre-FFN) + FFN multi + Residual Add → 1 ELF

This is the most impactful merge. Currently 3 invocations:
- rmsnorm (pre-FFN): ~6ms
- ffn_multi: ~50ms
- add (residual 2): ~5ms

All three have compatible shapes:
- rmsnorm: `(seq_len, emb_dim)` → `(seq_len, emb_dim)`
- ffn_multi: `(seq_len, emb_dim)` → `(seq_len, emb_dim)` (with internal intermediates)
- add: `(seq_len * emb_dim)` flat → `(seq_len * emb_dim)` flat

The RMSNorm has the weight broadcast issue at multi-tile, but at [1,1] it compiles fine. In a multi-launch ELF, each kernel is its own launch — the rmsnorm launch stays at [1,1] while FFN launches use [8,x].

```
func @ffn_block_full(
    %res1:       memref<2048x2048xbf16>,   # input (residual from attn)
    %ffn_norm_w: memref<2048xbf16>,        # FFN norm weight
    %normed2:    memref<2048x2048xbf16>,   # norm output → FFN input
    %w_gate:     memref<2048x8192xbf16>,
    %gate_buf:   memref<2048x8192xbf16>,
    %w_up:       memref<2048x8192xbf16>,
    %up_buf:     memref<2048x8192xbf16>,
    %swiglu_buf: memref<2048x8192xbf16>,
    %w_down:     memref<8192x2048xbf16>,
    %down_out:   memref<2048x2048xbf16>,   # FFN output
    %res1_flat:  memref<4194304xbf16>,     # residual input (flat alias of res1)
    %down_flat:  memref<4194304xbf16>,     # FFN output (flat alias of down_out)
    %output:     memref<4194304xbf16>,     # final output (flat)
):
    air.launch: RMSNorm [1,1]
    air.launch: Gate GEMM [8,4]
    air.launch: Up GEMM [8,4]
    air.launch: SwiGLU [8,1]
    air.launch: Down GEMM [8,4]
    air.launch: Residual Add [8,1]
    return
```

13 func args, 6 launches. Saves 2 XRT invocations.

**Estimated savings**: ~8-10ms (eliminate 2 dispatches + BO write/read overhead for all FFN weights)

But: the 2D/1D aliasing for residual add inputs is the same pattern we solved in FFN (collapse_shape). The weighted RMSNorm at [1,1] compiles fine as a single launch.

---

## Recommended Merge Order

| Priority | Merge | Invocations saved | Est. savings | Complexity |
|----------|-------|-------------------|-------------|-----------|
| **1** | **C: RMSNorm + FFN + Add** | 2 | ~8-10ms | Medium (6 launches, 13 args) |
| **2** | **A: RoPE Q + RoPE K** | 1 | ~2-3ms | Low (2 launches, 6 args) |
| **3** | **B: O GEMM + Add** | 1 | ~3-4ms | Medium (2D/1D aliasing) |

After all merges: **5 XRT invocations** per layer (down from 10):
1. RMSNorm (pre-attn) + attn_gemms → could merge too if we add rmsnorm launch
2. RoPE Q+K (merged)
3. FlashAttention
4. O GEMM + Residual Add (merged)
5. RMSNorm + FFN + Residual Add (merged)

---

## Implementation

### Task 1: Merge C — RMSNorm + FFN + Add

Extend `ffn_swiglu/run.py`'s stitching to include rmsnorm and eltwise_add launches. Build 3 separate kernel modules, stitch into 6-launch func.

### Task 2: Merge A — RoPE Q + K

Simple 2-launch stitch of two RoPE kernels with different sizes.

### Task 3: Merge B — O GEMM + Residual Add

2-launch stitch with 2D/1D collapse_shape for the add input.

### Task 4: Profile and compare

Run full 16-layer comparison.

---

## Expected Outcome

After all merges:
- Per-layer: ~140ms → ~125ms (estimated)
- 16 layers: ~2.45s → ~2.1s
- XRT invocations: 10 → 5 per layer
- Host overhead: ~17ms → ~8ms per layer
