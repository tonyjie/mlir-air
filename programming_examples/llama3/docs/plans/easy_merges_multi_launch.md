# Easy Merges Multi-Launch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-layer XRT invocations by merging non-transpose-dependent steps.

**Architecture:** Text-based MLIR stitching to merge adjacent kernel groups, ELF format.

**Status:** Merge A + B done (10 → 8 invocations). Merge C ready but not integrated.

---

## Current State (6 XRT invocations per layer) — After Merge A + B + C

```
1. rmsnorm (pre-attn)           ~8ms   xclbin
2. attn_gemms (Q+K+V)           ~8ms   ELF (3 launches)
   --- HOST TRANSPOSE ---
3. rope_qk (Q+K merged)         ~11ms  ELF (2 herds)         ← Merge A
   --- HOST TRANSPOSE ---
4. flash_attn                   ~22ms  ELF
   --- HOST TRANSPOSE ---
5. o_proj_add (O GEMM + Add)    ~7ms   ELF (2 launches)      ← Merge B
6. ffn_full (RMS+FFN+Add)       ~58ms  ELF (6 launches)      ← Merge C
```

**5 XRT invocations** (was 10). Total kernel: ~113ms/layer. 16 layers: 1.81s.

## Plan A (completed): RMSNorm + Attn GEMMs merge

Merged pre-attention RMSNorm into attn_gemms → `rms_attn_gemms` (4 launches). Reduced from 6 → 5 invocations.

## Plan B (investigated, deferred): DMA Transpose Launches

**Goal**: Merge rms_attn_gemms + transpose(Q,K,V) + rope_qk → 1 ELF, and transpose(attn_out) + o_proj_add → 1 ELF. Would reduce from 5 → 3 invocations.

**Blocker**: AIE DMA hardware requires **innermost stride = 1 for sub-32b datatypes** (BF16 is 16-bit). Pure DMA-based transpose (strided writes) only works for 32b+ types. See `data_transfer_transpose/dma/` (uint32, works) vs `data_transfer_transpose/dma_bf16/` (needs C++ kernel).

**What we tried**:
- Built `dma_transpose.py` with strided 2D DMA write pattern from `data_transfer_transpose/dma/transpose.py`
- Compilation fails: `'aie.dma_bd' op For <32b width datatypes, inner-most dim stride must be 1`
- Also tried 1D per-row DMA approach — compiles but produces incorrect results (compiler pass transforms loop-carried DMAs in unexpected ways)

**What would be needed**:
- A tiled C++ transpose kernel (`transpose_bf16`) that transposes a tile_m × tile_k block in L1
- An AIR launch wrapper that tiles the full matrix, DMA-loads each tile, calls the C++ kernel, DMA-writes back
- The existing `data_transfer_transpose/dma_bf16/transpose.cc` is a starting point but only handles one full matrix (not tiled)
- For LLAMA Q: (2048, 2048) at BF16 = 8MB, far exceeds L1 (64KB), so tiling is mandatory

**Estimated savings**: ~3-5ms/layer (CPU transposes currently ~2-3ms + dispatch overhead savings)

**Decision**: Deferred. Current 5-invocation pipeline at 113ms/layer is already 26% faster than IRON (152ms). The complexity of building and debugging a tiled BF16 transpose kernel doesn't justify ~3% improvement.

## Previous State (10 XRT invocations per layer) — Before Merge A + B

```
1. rmsnorm                      ~11ms  xclbin
2. attn_gemms (Q+K+V)           ~10ms  ELF (3 launches)
3. rope_q                       ~10ms  xclbin               ← now merged
4. rope_k                       ~3ms   xclbin               ← now merged
5. flash_attn                   ~22ms  ELF
6. gemm_qo (O proj)             ~8ms   xclbin               ← now merged
7. add (residual 1)              ~5ms   xclbin               ← now merged
8. rmsnorm (pre-FFN)             ~11ms  xclbin
9. ffn_multi                     ~67ms  ELF (4 launches)
10. add (residual 2)             ~5ms   xclbin
```

Total kernel: ~140ms/layer. 16 layers: 2.45s.

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

## Merge Status (updated 2026-03-31)

| Priority | Merge | Invocations saved | Status | File |
|----------|-------|-------------------|--------|------|
| **1** | **C: RMSNorm + FFN + Add** | 2 | **DONE** — integrated into LLAMA pipeline | `ffn_full_multi.py` |
| **2** | **A: RoPE Q + RoPE K** | 1 | **DONE** — integrated into LLAMA pipeline | `rope_qk_multi.py` |
| **3** | **B: O GEMM + Add** | 1 | **DONE** — integrated into LLAMA pipeline | `o_proj_add_multi.py` |

### Current State (Merge A + B integrated)

**XRT invocations per layer**: 8 (down from 10)

```
1. rmsnorm (pre-attn)                ~11ms   xclbin
2. attn_gemms (Q+K+V)               ~10ms   ELF (3 launches)
   --- HOST TRANSPOSE ---
3. rope_qk (Q+K merged)             ~13ms   ELF (2 herds)    ← Merge A
   --- HOST TRANSPOSE ---
4. flash_attn                        ~22ms   ELF (or CPU fallback)
   --- HOST TRANSPOSE ---
5. o_proj_add (O GEMM + Res Add)     ~9ms   ELF (2 launches) ← Merge B
6. rmsnorm (pre-FFN)                 ~11ms   xclbin
7. ffn_multi (Gate+Up+SiLU+Down)     ~67ms   ELF (4 launches)
8. add (residual 2)                   ~5ms   xclbin
```

**Per-layer kernel time**: ~130ms (was ~140ms, saved ~10ms from merge overhead)
**Total kernel time (16 layers)**: ~2.02s

### Profiled Results (16 layers, CPU attention)

| Kernel | Avg (ms) | Count/layer | Notes |
|--------|----------|-------------|-------|
| rmsnorm | 11 | x2 | xclbin, [1,1] herd |
| attn_gemms | 10 | x1 | ELF, 3 launches |
| rope_qk | 13 | x1 | ELF, 2 herds (was 2 separate: 10+3ms) |
| o_proj_add | 9 | x1 | ELF, 2 launches (was 2 separate: 8+5ms, **saved 4ms**) |
| ffn_multi | 67 | x1 | ELF, 4 launches |
| add | 5 | x1 | xclbin |
| **Total** | **~130** | **8 invocations** | Down from 10 |

### Remaining Merge: C (RMSNorm + FFN + Add)

Would reduce to 6 invocations. Currently blocked — the 6-launch module compiles and runs (corr=0.999) but needs full integration work. The `air.segment` wrapper fix resolved the compiler bug. Key files:
- `ffn_full_multi.py` — 6-launch builder (RMS + Gate + Up + SwiGLU + Down + Add)
- `repro_herd_load_bug.py` — Reproducer for the compiler bug (bare herd without segment)
