# Phase 1 — Per-kernel shape audit (Qwen2.5-1.5B)

**Date**: 2026-04-18
**Approach**: Source-side parametricity audit + variant audit (precedent set
by `llama32_3b`/`smollm2_1_7b` deployments). Standalone XRTRunner per-shape
kernels are deferred to Phase 2 single-block correctness; the per-kernel
verify table in Phase 2 functions as the PASS evidence for parametric items.

## Step 0 — Variant audit (LESSON 3 / `llama32_3b` 2026-04-18)

| Production code path | Python builder it imports | Lit-test coverage |
|---|---|---|
| `_llm_shared/phase_helpers/headfirst_fa.patch_run_cached_for_headfirst_fa` | `flash_attention/.../attn_npu2.py` (head-first) | `run_npu2_makefile_peano_llama3_8b.lit` exercises the same `attn_npu2.py` at `DK=128 DV=128` ✓ |
| `multi_launch_builder/rms_gemms_rope_multi.py` | inline IR (parametric) | covered by smollm2/llama32_3b Phase 2 ✓ |
| `multi_launch_builder/o_ffn_multi.py` | inline IR (parametric) | covered by smollm2/llama32_3b Phase 2 ✓ |
| `multi_launch_builder/lm_head_multi.py` (prefill) | inline IR + `n_partitions` | covered by llama3 (vocab=128256) ✓ |
| `multi_launch_builder/lm_head_gemv_multi.py` (decode) | inline IR + `n_partitions` | covered by llama3 ✓ |

No coverage gap for FA: head_dim=128 is the same exercise that
`llama32_3b/` validated; we will reuse the **Option C head-first wrapper**
unchanged.

## Shape enumeration (Qwen2.5-1.5B, seq_len=2048)

| Op | Shape (M, N, K) or other | Source vs prior models |
|---|---|---|
| RMSNorm | (2048, 1536) | NEW emb_dim |
| Q GEMM | (2048, 1536, 1536) | NEW (square 1536) |
| K GEMM | (2048, 256, 1536) | NEW narrow-N (KV dim = 2 × 128) |
| V GEMM | (2048, 256, 1536) | NEW narrow-N |
| O GEMM | (2048, 1536, 1536) | NEW (square 1536) |
| **QKV bias add** | (2048, 1536) for Q; (2048, 256) for K, V | **NEW kernel pattern** |
| RoPE | head_dim=128, (2048 × n_heads or n_kv_heads) rows | covered by `llama32_3b` |
| FlashAttention (head-first via Option C) | n_heads=12, n_kv_heads=2, lq=lk=2048, dk=dv=128, group=6 | NEW group=6 (`llama32_3b` was group=3) |
| Gate GEMM | (2048, 8960, 1536) | NEW hidden_dim |
| Up GEMM | (2048, 8960, 1536) | NEW hidden_dim |
| SiLU + mul | hidden_dim=8960 | NEW shape, parametric kernel |
| Down GEMM | (2048, 1536, 8960) | NEW K=8960 |
| Eltwise add (residual) | (2048, 1536) | parametric, drop-in |
| Final RMSNorm | (1536,) | drop-in |
| LM Head GEMM (prefill) | (2048, 151936, 1536) | **NEW vocab partition needed** |
| LM Head GEMV (decode) | (1, 151936, 1536) | **NEW vocab partition needed** |
| Q/K/V/O GEMV (decode) | M=1, K=1536, N ∈ {1536, 256} | NEW K=1536 |
| Down GEMV (decode) | M=1, K=8960, N=1536 | **NEW K=8960 → mv_k8960.o renamed kernel** |

## Classification

### Drop-in (no new code, parametric in existing builders)
- RMSNorm at emb_dim=1536 (1536 = 8 × 192 — divisible by RMSNorm's 8-tile herd_x ✓)
- Eltwise add (residual stream)
- RoPE half-split at head_dim=128 (parametric, validated in `llama32_3b/`)
- All Q/K/V/O GEMM/GEMV at K=1536 or N∈{1536, 256} via `gemm_builder` /
  `gemv` builders. **Caveat**: K=256 narrow-N may want smaller `tile_n`.
  Default `tile_n=64` divides 256 (4 tiles) — acceptable.
- SiLU + mul at hidden_dim=8960 (8960 = 64 × 140; default `swiglu_tile_n=4096`
  doesn't divide 8960 cleanly — needs `swiglu_tile_n=2240` or 4480).
- FlashAttention via Option C head-first wrapper (same builder + wrapper
  used in `llama32_3b/`); `compile_attn_npu2_split(lqp=256, lkp=64,
  dk=dv=128, num_q_tiles=4)` is the proven config.

### Recompile (parametric IR; unique config baked at compile time)
- `rms_gemms_rope_multi.build_rms_gemms_rope_module(emb_dim=1536,
  n_heads=12, n_kv_heads=2, head_dim=128, seq_len=2048)` — baked module per
  Qwen2.5 shapes. Multi-launch ELF will include 6 launches (RMSNorm + Q GEMM
  + K GEMM + V GEMM + RoPE Q + RoPE K).
- `o_ffn_multi.build_o_ffn_module(emb_dim=1536, hidden_dim=8960, ...)`. Need
  to retune the SwiGLU tile_n and possibly Down K_l2 (8960 % 256 = 0 ✓; OK).
- `o_gemv_ffn_multi.build_o_gemv_ffn_module(...)` for decode — needs
  re-tile-tuning for K=8960 (similar to llama3's mv_k8192 path).

### NEW work (genuinely novel for this deployment)

1. **QKV bias addition (Qwen2 architectural feature)** — required for
   correctness. **Plan**: easiest implementation is a separate eltwise-add
   pass between the GEMM and RoPE in the multi-launch builder. Three options:

   - **Option A (recommended)**: extend
     `rms_gemms_rope_multi.build_rms_gemms_rope_module` with a `qkv_bias` flag.
     When true, emit additional `air.launch` ops between `gemm_q -> rope_q`
     etc. that broadcast-add `bq/bk/bv` (1-D) to the GEMM 2-D output.
     ~2 hours; uses the existing `_build_add_2d_to_2d` helper (already in
     o_ffn_multi.py for residual add) — generalize to broadcast 1-D over the
     M axis.

   - **Option B**: pre-fold bias into a CPU-side prep step that prepends a
     bias-row to the input `x`. Requires changing the GEMM to compute
     `[1; x] @ [b; W]` and breaks the multi-launch design. Reject.

   - **Option C**: emit bias-add as a stitched-in pre-RoPE micro-launch via
     a separate multi-launch builder. Possible but worse than A because it
     needs a third `air.launch` per Q/K/V (= 3 extra launches = 9 in the
     stitched ELF instead of 6). Likely a small perf regression vs Option A's
     fused approach.

   Decision: **Option A** for Phase 2 implementation. Bias Σ-broadcast across
   M is a cheap kernel; same pattern as the residual eltwise add but
   broadcasting a 1-D operand.

2. **LM-Head partition design (vocab=151936)** — neither 8 × 16384 (=131072
   < 151936) nor 8 × 19200 (= 153600, requires re-tile-tune) fits cleanly.
   **Plan**: 10 partitions × 16384 = 163840, padded by 11904. Verify
   `lm_head_multi.build_lm_head_module(n_partitions=10)` accepts 10 (current
   default is 8 — should be parametric on n_partitions; minor IR-emit fix
   if not). Same partition scheme for prefill (`lm_head_multi`) and decode
   (`lm_head_gemv_multi`). ~30 min if `n_partitions` is parametric.

3. **`mv_k8960.o` renamed external kernel** — Down GEMV (decode) at K=8960
   needs a per-K rename (analogous to `mv_k8192.o` for llama3 K=8192) so it
   can coexist with K=1536 GEMVs in the same ELF. **Plan**: copy the
   `_ensure_mv_k8192_o` codepath from `llama3_decode.py` and parameterize on
   K. ~30 min.

### Risk surfaced for Phase 2

- **NPU FA at GQA group=6 is untested.** `llama32_3b/` exercised group=3.
  Group is a software replication factor over KV heads in the head-first FA
  Python IR; should be parametric, but `n_kv_heads=2` is the smallest yet
  (could expose herd-shape edge cases — most prior tests used n_kv_heads ≥ 8
  with FA herd_x = 8). If Phase 2 single-block FA fails, follow the
  `debug-fa-runtime-failure` recipe with a targeted bisect on
  `(n_heads=12, n_kv_heads=2, lq=lk=2048, dk=dv=128)`.

- **`emb_dim=1536` is the smallest seen.** RMSNorm 8-tile herd_x assumes
  emb_dim divisible by 8 × tile_size. 1536 / 8 = 192 — works.
  o_ffn's `o_tile_k_l2=256` — 1536 / 256 = 6, works. Most defaults appear
  fine but tile-alignment edge cases will surface in Phase 2 compile.

## Status table

| Kernel | Shape | Status | Recovered via / Plan |
|---|---|---|---|
| RMSNorm | (2048, 1536) | DROP-IN | parametric, validated in Phase 2 |
| Q GEMM | (2048, 1536, 1536) | RECOMPILE | parametric |
| K GEMM | (2048, 256, 1536) | RECOMPILE | parametric (verify narrow-N tiles) |
| V GEMM | (2048, 256, 1536) | RECOMPILE | parametric |
| **QKV bias add** | (2048, 1536) / (2048, 256) | **NEW** | **Option A — extend rms_gemms_rope_multi** |
| RoPE | head_dim=128 | DROP-IN | inherited from llama32_3b |
| FlashAttention | head-first, group=6, dk=128 | RECOMPILE + RISK | Option C wrapper; bisect if fail |
| O GEMM | (2048, 1536, 1536) | RECOMPILE | parametric |
| Gate/Up GEMM | (2048, 8960, 1536) | RECOMPILE | retune `gate_tile_n`, `swiglu_tile_n` |
| SiLU+mul | hidden_dim=8960 | RECOMPILE | parametric |
| Down GEMM | (2048, 1536, 8960) | RECOMPILE | parametric |
| Final RMSNorm | (1536,) | DROP-IN | parametric |
| **LM Head GEMM** | (2048, 151936, 1536) | **NEW partition** | 10 × 16384 partition |
| Q/K/V/O GEMV (decode) | K=1536 | RECOMPILE | parametric |
| **Down GEMV (decode)** | K=8960 | **NEW** | **`mv_k8960.o` renamed kernel** |
| **LM Head GEMV** | (1, 151936, 1536) | **NEW partition** | 10 × 16384 partition |

## Phase 1 PASS criteria

- All shapes are either DROP-IN or RECOMPILE off existing parametric builders ✓
- Three NEW work items (QKV bias, LM-head partition, mv_k8960) have explicit plans ✓
- Variant audit confirms FA path uses the lit-tested `attn_npu2.py` ✓
- One RISK surfaced (FA group=6) with the debug-fa-runtime-failure recipe ready ✓

**Phase 1 PASSES** with the three NEW items and one risk recorded for Phase 2.
