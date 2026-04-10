# Decode Multi-Launch Merging Plan

## Current State (3 XRT invocations per block — DEPLOYED)

```
---- Pre-attention (1 XRT call) ----
1. rms_gemv_rope (6 launches)        0.9ms/block   ELF
---- CPU attention (~0.3ms) ----
2. o_gemv_ffn (8 launches)           3.7ms/block   ELF
---- After 16 blocks ----
3. lm_head_gemv (8 launches)         13.5ms         ELF (NPU, was CPU 258ms)
```

**Total: 92ms/token (10.8 tok/s)** — 4.0x faster than IRON (370ms)

---

## Previous State (10 XRT invocations per block)

```
---- Pre-attention ----
1.  rmsnorm (pre-attn)       [1,1]    0.3ms   xclbin
2.  qkv_gemv (Q+K+V)        [8,1]×3  1.0ms   ELF (3 launches)
3.  rope_q                   [1,1]    0.3ms   xclbin
4.  rope_k                   [1,1]    0.2ms   xclbin
---- CPU attention (~2ms) ----
5.  o_gemv_add (O+Add)       [8,1]×2  0.6ms   ELF (2 launches)
6.  rmsnorm (pre-FFN)        [1,1]    0.3ms   xclbin
7.  gate_up_gemv (Gate+Up)   [8,1]×2  2.5ms   ELF (2 launches)
8.  silu_mul                 [8,1]    0.3ms   xclbin
9.  gemv_down                [8,1]    2.1ms   xclbin
10. add                      [8,1]    0.3ms   xclbin
```

**Total: 10 XRT calls/block, ~8ms kernel time, 351ms/token (16 layers)**

## Why Decode Can Merge More Than Prefill

Prefill's merging was blocked by FlashAttention's channel/cascade routing complexity
in the `aircc` compiler. Decode uses **CPU attention** (no NPU FlashAttention), so
this blocker doesn't apply. All decode kernels are simple GEMV/RMSNorm/RoPE/eltwise
ops with [1,1] or [8,1] herds — no channels, no cascades.

## Proposed Merges

### Merge A: Pre-attention block → `rms_gemv_rope` (6 launches)

Merge steps 1-4:
```
L1: RMSNorm      [1,1]   x × norm_w → normed         (1 launch)
L2: Q GEMV       [8,1]   normed × wq_t → q            (1 launch)
L3: K GEMV       [8,1]   normed × wk_t → k            (1 launch)
L4: V GEMV       [8,1]   normed × wv_t → v            (1 launch)
L5: RoPE Q       [1,1]   q × lut_q → q_roped          (1 launch)
L6: RoPE K       [1,1]   k × lut_k → k_roped          (1 launch)
```

**6 launches** — same as prefill's `rms_gemms_rope`. Proven compile range (33s for prefill).

Notes:
- RMSNorm: [1,1] herd, M=1 (single token), N=2048
- GEMV: [8,1] herd, 8-column parallel. Uses transposed weights (K×N layout)
- RoPE Q: [1,1] herd, 32 rows × 64 cols (n_heads × head_dim)
- RoPE K: [1,1] herd, 8 rows × 64 cols (n_kv_heads × head_dim)
- Shape aliasing: GEMV outputs (2048,) or (512,) 1D; RoPE also 1D — no collapse_shape needed

Estimated args: normed(2048), norm_w(2048), wq_t(2048×2048), wk_t(512×2048), wv_t(512×2048),
q(2048), k(512), v(512), lut_q(2048), lut_k(512), q_roped(2048), k_roped(512) = ~12 args

### Merge B: Post-attention + FFN → `o_gemv_ffn` (8 launches)

Merge steps 5-10:
```
L1: O GEMV       [8,1]   attn_out × wo_t → proj        (1 launch)
L2: Add          [8,1]   proj + x_residual → res1       (1 launch)
L3: RMSNorm      [1,1]   res1 × ffn_norm_w → normed2   (1 launch)
L4: Gate GEMV    [8,1]   normed2 × wgate_t → gate      (1 launch)
L5: Up GEMV      [8,1]   normed2 × wup_t → up          (1 launch)
L6: SiLU×mul     [8,1]   SiLU(gate) × up → swiglu      (1 launch)
L7: Down GEMV    [8,1]   swiglu × wdown_t → down       (1 launch)
L8: Add          [8,1]   down + res1 → output           (1 launch)
```

**8 launches** — same as prefill's `o_ffn` and LM Head. Proven compile range (50-108s).

Notes:
- All herds are [8,1] or [1,1] — simple routing
- res1 shared between L2 output and L3/L8 input (same 2D→1D pattern as prefill's o_ffn)
- Down GEMV has K=8192 (different from others K=2048) — different backend flags needed?
  Currently `gemv_down` uses `_GEMV_K8192_BACKEND` while others use `_GEMV_K2048_BACKEND`
  In a merged module, all launches compile with the same backend flags — need to verify this works

Estimated args: attn_out(2048), wo_t(2048×2048), proj(2048), x_residual(2048), res1(2048),
ffn_norm_w(2048), normed2(2048), wgate_t(8192×2048), gate(8192), wup_t(8192×2048),
up(8192), swiglu(8192), wdown_t(2048×8192), down(2048), output(2048) = ~15 args

## Results

### Merge A: COMPLETE

`rms_gemv_rope_multi.py` — 6 launches, 13 args. Compiles in **3.2s**, PASS (corr=0.999979).

### Merge B: COMPLETE (blocker resolved)

`o_gemv_ffn_multi.py` — 8 launches, 15 args. Compiles in **10.9s**, PASS (corr=0.998789).

**Solution for K-dimension mismatch**: Renamed the Down GEMV's external kernel function
via C++ preprocessor `-D` defines during `.o` compilation:
- `mv.o`: exports `matvec_vectorized_bf16_bf16` (K=2048 GEMVs)
- `mv_k8192.o`: exports `dg_matvec_vectorized_bf16_bf16` (K=8192 Down GEMV)
Both `.o` files link to different cores in the same ELF. The MLIR module has two separate
`func.func private` declarations with different names, types, and `link_with` attributes.

### Previous blocker (now resolved): external kernel type mismatch

The `@matvec_vectorized_bf16_bf16` external C++ kernel has a fixed signature that includes
the K dimension in its memref types. K=2048 GEMVs and K=8192 Down GEMV produce different
function signatures:
- K=2048: `@matvec_vectorized_bf16_bf16(i32, i32, i32, memref<4x2048xbf16, 2>, memref<2048xbf16, 2>, memref<8xbf16, 2>)`
- K=8192: `@matvec_vectorized_bf16_bf16(i32, i32, i32, memref<1x8192xbf16, 2>, memref<8192xbf16, 2>, memref<2xbf16, 2>)`

MLIR requires a single `func.func private @name(...)` declaration per module. Two calls
with different signatures to the same name is a parse error.

**This is the "memref type mismatch between GEMVs with different tile_m" blocker** noted
in LLAMA_PLAN.md. It applies to ANY multi-launch merge combining K=2048 and K=8192 GEMVs.

### Target: 2 XRT calls/block (ACHIEVED)

```
---- Pre-attention (1 XRT call) ----
1. rms_gemv_rope (6 launches)        ~2ms    ELF   ← Merge A (DONE)
---- CPU attention (~2ms) ----
2. o_gemv_ffn (8 launches)           ~6ms    ELF   ← Merge B (DONE)
```

**2 XRT calls/block** (down from 10). Both merges integrated into `llama3_decode.py`.

### Performance Analysis (same-session benchmarks, 3 runs each, tokens 5-9 avg)

| Config | XRT calls | Speed (avg ± range) |
|--------|-----------|---------------------|
| 10-call original | 10 | 407ms (398-413) |
| **2-call merged** (tile_m=8, omit_pingpong=all, intermediate_indices) | **2** | **369ms (366-373)** |

**The 2-call merged pipeline is 9-10% faster** (~38ms/token savings).

Note: The previous "351ms/token" baseline was from a different session and is not
reproducible under the same conditions. Same-session comparison is the only reliable method.

**Why the merge wins despite omit_pingpong penalty:**
- 8 fewer XRT dispatch calls per block (~8 × 0.1ms = ~0.8ms saved)
- Intermediate buffers stay on device: no redundant host↔device transfers for
  normed, q, k, v, proj, res1, normed2, gate, up, swiglu, down (~10 fewer BO syncs)
- `intermediate_indices` skips unnecessary zero-buffer writes
- Combined savings (~38ms) exceed the omit_pingpong cost (~19ms)

**Root causes investigated:**

| Factor | Impact | Status |
|--------|--------|--------|
| `tile_m=2` for K=2048 GEMVs | +14ms/token | **FIXED** — restored tile_m=8 after rename fix |
| `omit_pingpong="all"` for K=2048 GEMVs | +19ms/token | Required by K=8192 Down GEMV |
| Eliminated intermediate BO syncs | -38ms/token | **NEW** — intermediates stay on device |
| `intermediate_indices` optimization | -8ms/token | **NEW** — skips zero-buffer writes |

## Workarounds for Full Merge B (attempted and future)

1. **Rename external kernel via -D defines** (SOLVED):
   - Renamed Down GEMV's `@matvec_vectorized_bf16_bf16` to `@dg_matvec_vectorized_bf16_bf16`
   - Changed `link_with` from `"mv.o"` to `"mv_k8192.o"` in both func decl and herd
   - Compiled `mv_k8192.o` with `-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16`
   - Pre-placed `mv_k8192.o` in `build_peano/` and `air_project/` with correct symbols
   - `aiecc` finds the `.o` via its CWD search, copies to tmpDir, linker script has `INPUT(mv_k8192.o)`
   - **Result**: 8-launch module compiles in 10.9s, PASS (corr=0.998789)
2. **Template-style external kernel**: The `.o` file could export multiple entry points
   parameterized by K dimension. Requires modifying the C++ kernel source.
3. **Inline GEMV**: Implement GEMV computation inline in MLIR (vectorized matmul).
   Avoids external function signature issue entirely. Most promising long-term solution.

## Implementation Priority

1. Start with Merge B (steps 5-10) — more XRT calls saved (6→1), bigger impact
2. Then Merge A (steps 1-4) — 4→1 XRT calls saved
3. Verify with `--verify` and profile with `--profile`

## Files

| File | Action |
|------|--------|
| `multi_launch_builder/o_gemv_ffn_multi.py` | CREATE — Merge B builder |
| `multi_launch_builder/rms_gemv_rope_multi.py` | CREATE — Merge A builder |
| `llama3_decode.py` | MODIFY — integrate merged kernels |
| `multi_launch_builder/rms_qkv_gemv_multi.py` | READ — existing QKV GEMV pattern |
| `multi_launch_builder/o_gemv_add_multi.py` | READ — existing O+Add pattern |
| `multi_launch_builder/ffn_gemv_multi.py` | READ — existing Gate+Up pattern |
