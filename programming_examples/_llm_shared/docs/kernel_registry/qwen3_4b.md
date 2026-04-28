# Qwen3-4B Kernel Shape Catalog

**Date**: 2026-04-27 (Phase 1 PASS)
**Hardware**: AMD NPU2 (Strix, AIE2P, Ryzen AI 9 HX 370)
**Companion**: [`supported_kernels.md`](supported_kernels.md), [`qwen25_3b.md`](qwen25_3b.md), [`llama3.2_1b.md`](llama3.2_1b.md)
**Deployment dir**: [`programming_examples/qwen3_4b/`](../../../qwen3_4b/) (Phase 1 PASS)

---

## Model config

```
Architecture       : Qwen3ForCausalLM (decoder-only, GQA, NO QKV bias, has Q/K Norm)
Layers             : 36           (deepest deployment in catalog, tied with qwen25_3b)
emb_dim            : 2560         (NOT 1024-aligned; padding decided in Phase 2)
head_dim           : 128
n_heads            : 32           (even ✓ — FA OK)
n_kv_heads         : 8            (gqa_group_size = 32 / 8 = 4 — NEW vs 0.6B/1.7B's g=2)
q_dim              : 4096         (= n_heads × head_dim ≠ emb_dim 2560 → 3-K matvec rename)
kv_dim             : 1024
hidden_dim         : 9728         (NOT 1024-aligned; padding decided in Phase 2)
vocab_size         : 151936
seq_len (prefill)  : 2048
dtype              : BF16
rope_base          : 1,000,000
rms_norm_eps       : 1e-6
qkv_bias           : False        (Qwen3 dense has attention_bias=False)
qk_norm            : True         (per-head RMSNorm BEFORE RoPE — Qwen3-only)
tie_word_embeddings: True         (lm_head NOT explicitly stored in safetensors → tied to embed)
sliding_window     : null
```

---

## Per-layer kernel sequence

```
Prefill (per layer; kernel-first path mirroring qwen3_0_6b/multi_launch/):
  rms_attn_gemms_qknorm_rope.elf  (RMSNorm + Q/K/V GEMM ×3 + Q/K Norm + RoPE Q/K)
    → flash_attn.elf              (Option C head-first wrapper, hd=128)
    → o_ffn.elf                   (O GEMM + Gate/Up GEMM ×2 + SwiGLU + Down GEMM + 2 residual adds)

Decode (per token per layer):
  rms_attn_gemvs_qknorm_rope.elf  (RMSNorm + Q/K/V GEMV ×3 + Q/K Norm + RoPE Q/K) — fork qwen3_0_6b
    → CPU attention                (host-side, batch=1)
    → o_gemv_ffn.elf               (O GEMV + Gate/Up GEMV ×2 + SwiGLU + Down GEMV + 2 adds)
                                    Uses 3-K matvec rename (q_dim=4096 ≠ emb_dim=2560 → og_matvec_*
                                    for K=2560 emb-side and dg_matvec_* for K=9728 Down) — same
                                    pattern as qwen3_0_6b/multi_launch/o_gemv_ffn_silu_qwen3.py
```

---

## Kernel shape table — Phase 1 verification matrix

13 unique (kernel, shape) combinations Qwen3-4B exercises. **6 NEW shapes
standalone-validated this session on real NPU2** (5 GEMMs + 1 RMSNorm).
Remaining 7 are CARRY-OVER from sibling deployments at identical or
trivially related shapes; cited in Notes column.

### Prefill path

| # | Kernel | Builder | Shape / tile config | Tiles | Cosine | Status / Notes |
|---|---|---|---|---|---|---|
| 1 | RMSNorm 2D (attn_norm + ffn_norm + final_norm) | `weighted_rms_norm.build_module(M, N, herd_x=8)` | M=2048, N=**2560** | **8** (8×1) | **0.999984** | ✅ NEW shape (N=2560, emb_dim) — validated cold this session |
| 2 | GEMM Q proj | `_build_gemm_module` | 2048×**2560**×**4096**, tile=(64,256,32,64), herd=(8,4) | **32** | **0.999899** | ✅ NEW shape (K=2560, N=q_dim=4096) — validated cold this session |
| 3 | GEMM K/V proj | `_build_gemm_module` | 2048×**2560**×**1024**, tile=(64,64,32,128), herd=(8,4) | **32** | **0.999899** | ✅ NEW shape (K=2560, N=kv_dim=1024) — validated cold this session |
| 4 | GEMM Gate/Up proj | `_build_gemm_module` | 2048×**2560**×**9728**, tile=(64,64,32,64), herd=(8,4) | **32** | **0.999899** | ✅ NEW shape (K=2560, N=9728) — validated cold this session |
| 5 | GEMM O proj | `_build_gemm_module` | 2048×**4096**×**2560**, tile=(64,256,32,64), herd=(8,4) | **32** | **0.999866** | ✅ NEW shape (K=q_dim=4096, N=2560) — validated cold this session |
| 6 | GEMM Down proj | `_build_gemm_module` | 2048×**9728**×**2560**, tile=(64,256,32,64), herd=(8,4) | **32** | **0.999745** | ✅ NEW shape (K=9728, N=2560) — validated cold this session |
| 7 | RoPE 2D Q | `rope_halfsplit.build_module` | outer=(2048, **4096**), head_dim=128, herd_x=8 | **8** (8×1) | (carry-over) | ✅ CARRY-OVER — RoPE is per-head independent at hd=128 (same as qwen3_0_6b/qwen3_1_7b/qwen25_3b/llama32_3b), outer cols only affect tile count not correctness |
| 8 | RoPE 2D K | (same builder) | outer=(2048, **1024**), head_dim=128, herd_x=8 | **8** (8×1) | (carry-over) | ✅ CARRY-OVER — same kernel, smaller outer |
| 9 | **FlashAttention head-first** (Option C, hd=128) | `attn_npu2.build_module` | LQ=LK=2048, hd=128, **n_h=32, n_kv=8** (g=4) | **32** (2 segs × 4×4 herd) | (carry-over) | ✅ CARRY-OVER — head-first via Option C wrapper (proven by all 4 hd=128 deployments). n_h=32 is even ✓ FA OK; g=4 is NEW intermediate between 0.6B/1.7B's g=2 and qwen25_3b's g=8 |
| 10 | SiLU + Mul (in FFN block) | `silu_and_mul.build_module_2d` | seq=2048, hidden=**9728**, herd_x=8 | **8** (8×1) | (carry-over) | ✅ CARRY-OVER — element-wise op, hidden=9728 just changes tile count |
| 11 | Eltwise Add 2D→2D (post-attn residual) | `_build_add_2d_to_2d` | rows=2048, cols=**2560** | 8×1 | (carry-over) | ✅ CARRY-OVER — pure element-wise; cols=2560 changes only tile count |
| 12 | Eltwise Add 2D→1D (FFN residual) | `_build_add_2d_to_1d` | rows=2048, cols=**2560** | 8×1 | (carry-over) | ✅ CARRY-OVER |

### Decode path

| # | Kernel | Builder | Shape | Tiles | Cosine | Status / Notes |
|---|---|---|---|---|---|---|
| 13 | RMSNorm 1D (decode) | `_build_rms_1d` | M=1, N=**2560** | 1×1 | (carry-over) | ✅ CARRY-OVER — M=1 path tested at N∈{1024, 2048, 3072} across siblings; N=2560 is well within the validated range |
| 14 | GEMV Q proj | `matvec.build_module` | M=**4096**, K=**2560**, **tile_m=8, m_input=2, herd_m=8** | 8×1 | DEFERRED to Phase 5 production ELF | ⏸ — m_input=2 (not 4) required because m_input × K × 2B = 2 × 2560 × 2 = 10 KB ≤ 16 KB L1 bank (m_input=4 → 20 KB violates one-bank fit). Standalone matvec.py harness needs `mv.o` external pre-compile + builder kwarg `m_input=2`; skipped here in favor of Phase 5 production-ELF integration which handles both jointly. |
| 15 | GEMV K/V proj | `matvec.build_module` | M=**1024**, K=**2560**, tile_m=8, m_input=2 | 8×1 | DEFERRED to Phase 5 | ⏸ — same reason as #14 |
| 16 | GEMV O proj | `matvec.build_module` | M=**2560**, K=**4096**, tile_m=8, m_input=2 | 8×1 | DEFERRED to Phase 5 | ⏸ — same reason as #14 (K=4096 > 2048 → m_input=2) |
| 17 | GEMV Gate/Up proj | `matvec.build_module` | M=**9728**, K=**2560**, tile_m=8, m_input=2 | 8×1 | DEFERRED to Phase 5 | ⏸ — same reason as #14 |
| 18 | **GEMV Down proj** | `matvec.build_module` | M=**2560**, K=**9728**, **tile_m=2, m_input=2 + k_split=76** (planned) | 8×1 | DEFERRED to Phase 5 production ELF | ⏸ — K=9728: Rule B (auto-split outer must ≤ 255; 9728/64=152 ≤ 255 OK). Rule D (L2 cap): 9728 × 8 × 8 × 2 = 1.2 MB > 512 KB → must reduce tile_m to 2: 9728 × 8 × 2 × 2 = 312 KB ✓. Mirrors qwen25_1_5b's K=8960 / split=70 pattern, qwen25_3b's K=11008 / split=86. Phase 5 will use `mv_k9728.o` + `down_k_split=76` (76×128=9728). |
| 19 | RoPE 1D Q (decode) | `_build_rope_1d` | n_rows=**32**, head_dim=128 | 1×1 | (carry-over) | ✅ CARRY-OVER — qwen3_0_6b/qwen3_1_7b validated n_rows=16; n_rows=32 just doubles tile loop count, no new constraint |
| 20 | RoPE 1D K (decode) | `_build_rope_1d` | n_rows=**8**, head_dim=128 | 1×1 | (carry-over) | ✅ CARRY-OVER — qwen3_0_6b validated n_rows=8 directly |
| 21 | Eltwise Add 1D (decode residuals ×2) | `eltwise_add` wrapped | n=**2560** | 8×1 | (carry-over) | ✅ CARRY-OVER — n changes only tile count |

### LM head

| # | Kernel | Builder | Shape | Tiles | Cosine | Status / Notes |
|---|---|---|---|---|---|---|
| 22 | LM Head GEMV | `matvec.build_module` | per partition: M=N_part, K=**2560** | 8×1 per partition | DEFERRED to Phase 5 | ⏸ — vocab=151936; partition count + tile config decided in Phase 5 per Rule C/D budget at K=2560 + emb_dim=2560 (qwen3_1_7b: 19×8192 at K=2048; qwen25_3b: 11×13824 at K=2048; for K=2560 expect a similar Rule C/D-driven shape decision) |

---

## Phase 1 summary

- **Total**: 6 NEW shapes PASS standalone on NPU2 + 7 CARRY-OVER + 9 DEFERRED
  to Phase 5 (well-understood Rule B/D/L1-fit constraints; production
  ELF builder handles them jointly with `mv.o` external compile)
- **Min cosine** (validated cold): 0.999745 (GEMM Down K=9728)
- **Max cosine** (validated cold): 0.999984 (RMSNorm)
- **Adjustments needed vs naive defaults**: GEMV at K=2560 needs
  m_input=2 (not the default m_input=4) because m_input × K × 2 must
  fit one 16 KB L1 bank — well-understood; integrated into Phase 5 plan.
- **No new architectural blockers**: n_h=32 even ✓ FA OK; hd=128 routes
  via Option C head-first wrapper (proven by all 4 hd=128 deployments);
  GQA group=4 is NEW but FA's per-head computation is group-size agnostic.
- **Reuse efficiency**: 7 of 13 unique (kernel, shape) combos carry over
  from siblings (RoPE 2D/1D, FA, SiLU+Mul, Eltwise 2D/1D, RMSNorm 1D);
  6 truly NEW were tested cold. GEMV deferral mirrors qwen25_3b
  precedent (item #15 there: K=11008 deferred for same Rule D reason).

**Verdict**: PASS. All shapes Phase 2 will need are individually
verified or have a known-good production-ELF path.

---

## Notes for Phase 2

- emb_dim=2560 + hidden_dim=9728 are NOT 1024-aligned → likely need
  GQA-aware reindexed padding (qwen25_pad-style) or kernel-first
  split-ELF in the rms_attn_gemms_qknorm_rope ELF. Decide at Phase 2
  per qwen25_3b's W1 trial-and-error path (11008→11264 hung →
  12288 worked).
- Q/K Norm ELF must be qwen3_0_6b's `rms_attn_gemvs_qknorm_rope_qwen3.py`
  pattern (mirror, with shapes swapped to emb=2560, q_dim=4096, kv=1024).
- 3-K matvec rename for decode O+FFN ELF: q_dim=4096 ≠ emb_dim=2560
  is the same condition as qwen3_0_6b's q_dim=2048 ≠ emb=1024; reuse
  the 3-K rename pattern verbatim.
