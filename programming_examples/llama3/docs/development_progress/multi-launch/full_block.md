# Full Transformer Block Multi-Launch — Plan & Status

**Goal:** Reduce per-layer XRT invocations by merging kernels into larger multi-launch ELFs, ultimately targeting a single XRT invocation per transformer block.

**Key enabler:** Seq-first layout eliminates all host-side transposes between kernels, making data flow compatible between all launches within a single ELF.

---

## Current State (3 XRT invocations per layer) — Phase A + D complete

```
1. rms_gemms_rope             ~8ms    ELF (6 launches: RMS+Q+K+V+RoPE_Q+RoPE_K)
2. flash_attn                 ~22ms   ELF (1 launch)
3. o_ffn                      ~41ms   ELF (8 launches: O+Add+RMS+Gate+Up+SiLU+Down+Add)
```

**Per-layer kernel time**: ~71ms (with weight pre-loading, 4% BO overhead)
**Total prefill (16 layers)**: **1.30s kernel / 1.54s wall** (was 1.77s → 1.49s → 1.25s → 1.30s kernel)
**vs IRON**: 1.30s (kernel) vs 2.744s = **2.1x faster**

This is the **theoretical minimum** given FlashAttention's incompatibility with multi-launch merging.

---

## Phase A: RMSNorm + QKV GEMMs + RoPE Q+K → `rms_gemms_rope` (COMPLETE)

Merged `rms_attn_gemms` (4 launches) + `rope_qk` (2 herds) into a single 6-launch ELF.

**Key technical achievements:**
- **2D→1D collapse_shape inside launch**: GEMM outputs 2D `(2048, 2048)`, RoPE needs 1D `(4194304,)`. `memref.collapse_shape` inside the RoPE launch body resolves the type mismatch.
- **Outer 2D shape decoupling**: RoPE's outer memref type `(2048, 2048)` matches GEMM for arg sharing, but internally processes `65536 × 64` (n_heads × seq_len rows of head_dim width).
- **RoPE herd wrapping**: RoPE herds wrapped in `air.launch { air.segment { air.herd } }` to avoid `airrt-to-npu` legalization failure when mixed with segment-based GEMM launches.
- **13 func args, 6 launches**: Compiles in ~33s, runs correctly (corr=0.999 standalone, top-1 " Paris" in LLAMA).

**Files:**
- `multi_launch_builder/rms_gemms_rope_multi.py` — 6-launch builder
- `llama3_prefill.py` — integrated (replaced rms_attn_gemms + rope_qk)

---

## Phase D: Merge o_proj_add + ffn_full → `o_ffn_multi.py` (COMPLETE)

Merged O GEMM + Residual Add (2 launches) + FFN Full (6 launches) into a single 8-launch ELF.

**Key technical achievement:**
- **2D→2D collapse_shape for res1**: The residual add output (res1) feeds FFN RMSNorm which expects 2D. Used a new `_build_add_2d_to_2d` pattern where ALL 3 args (proj, x_residual, res1) are 2D memrefs with `collapse_shape` to 1D inside the launch. The next launch reads res1 as 2D directly — same bytes in DDR.
- **15 func args, 8 launches**: Compiles in ~50s.

**Files:**
- `multi_launch_builder/o_ffn_multi.py` — 8-launch builder
- `llama3_prefill.py` — integrated (replaced o_proj_add + ffn_full)

---

## Phase B: Add FlashAttention + O proj → `attn_half_multi.py` (BLOCKED)

Would merge: rms_gemms_rope + flash_attn + o_proj_add → 9 launches, 17 args.
- MLIR generates and parses correctly (1689 lines)
- **Blocked**: `aircc` compilation exceeds 10 minutes. Routing complexity scales super-linearly beyond 8 launches.
- See `compiler_scaling_analysis.md` for details.

## Phase C: Full Block → `transformer_block_multi.py` (BLOCKED)

Would merge all 15 launches into 1 ELF → **1 invocation/layer**
- MLIR generates and parses correctly (2355 lines, 27 args)
- **Blocked**: `aircc` compilation exceeds 1 hour. Same scaling issue as Phase B.
- `ulimit -s unlimited` fixes the `AIRDependencyCanonicalize` stack overflow, but compilation still too slow.
- Per-tile ELF compilation (256 tiles) completes in ~2 min; bottleneck is routing/instruction generation.

## Compiler Scaling Analysis

| Launches | Time | Practical? |
|----------|------|-----------|
| 6 | 33s | Yes (production) |
| 8 | 108s | Yes (LM Head, one-time) |
| 9 | >10min | No |
| 15 | >1hr | No |

See `compiler_scaling_analysis.md` for full details.

---

---

## Decode Multi-Launch (COMPLETE)

Decode merged to **3 XRT invocations per block** (down from 10) + NPU LM Head:

```
1. rms_gemv_rope    (6 launches)   0.9ms/block
---- CPU attention (~0.3ms) ----
2. o_gemv_ffn       (8 launches)   3.7ms/block
---- After 16 blocks ----
3. lm_head_gemv     (8 launches)   13.5ms (was CPU 258ms)
```

**Result: 91ms/token (9.5 tok/s)** — 4.1× faster than IRON (370ms).
See `decode_multi_launch.md` for details.

---

## Previous State History

### Before Phase D (4 invocations/layer, 1.71s)
```
1. rms_gemms_rope             ~11ms   ELF (6 launches)
2. flash_attn                 ~23ms   ELF (1 launch)
3. o_proj_add                  ~6ms   ELF (2 launches)
4. ffn_full                   ~52ms   ELF (6 launches)
```

### Before Phase A (5 invocations/layer, 1.77s)
```
1. rms_attn_gemms             ~9ms   ELF (4 launches)
2. rope_qk                    ~4ms   ELF (2 herds)
3. flash_attn                ~20ms   ELF (1 launch)
4. o_proj_add                 ~6ms   ELF (2 launches)
5. ffn_full                  ~52ms   ELF (6 launches)
```

### Before seq-first (10 invocations/layer, ~2.4s)
Host transposes between GEMM→RoPE→FlashAttn→O_GEMM. See `full_block_transpose_analysis.md` for the (now resolved) analysis.
