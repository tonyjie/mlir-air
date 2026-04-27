# SmolLM2-135M Kernel Shape Catalog

**Date**: 2026-04-27 (Phase 1 in progress)
**Hardware**: AMD NPU2 (Strix, AIE2P, Ryzen AI 9 HX 370)
**Companion**: [`supported_kernels.md`](supported_kernels.md), [`llama3.2_1b.md`](llama3.2_1b.md)
**Deployment dir**: [`programming_examples/smollm2_135m/`](../../../smollm2_135m/) (Phase 1 active)

---

## What this doc is

For SmolLM2-135M specifically: every leaf-kernel × shape the production
deployment will invoke, plus measured correctness + perf from real NPU2
runs. Empty rows are pending Phase 1 completion.

---

## Model config

```
Architecture       : LlamaForCausalLM (decoder-only, GQA)
Layers             : 30        (deepest deployment so far)
emb_dim            : 576
head_dim           : 64
n_heads            : 9         (ODD, non-power-of-2 — first deployment with this)
n_kv_heads         : 3         (gqa_group_size = 9 / 3 = 3)
q_dim              : 576       (= n_heads × head_dim = emb_dim)
kv_dim             : 192       (= n_kv_heads × head_dim)
hidden_dim         : 1536      (FFN intermediate; smallest deployment)
vocab_size         : 49152     (same as smollm2_1_7b)
seq_len (prefill)  : 2048
dtype              : BF16
rope_base          : 100000
tied_embeddings    : true      (lm_head shares embed_tokens.weight)
```

---

## Per-layer kernel sequence

Same as llama3 inheritance path (no Q/K Norm, no QKV bias):

```
Prefill (per layer, 3 XRT calls):
  rms_gemms_rope.elf  (RMSNorm + Q/K/V GEMM ×3 + RoPE Q/K)
    → flash_attn.elf  (seq-first FA — BLOCKED at n_h=9; see Active blockers)
    → o_ffn.elf       (O GEMM + Gate/Up GEMM ×2 + SwiGLU + Down GEMM + 2 residual adds)

Decode (per token per layer, 3 XRT calls):
  rms_gemv_rope.elf   (RMSNorm + Q/K/V GEMV ×3 + RoPE Q/K)
    → CPU attention   (host-side, batch=1)
    → o_gemv_ffn.elf  (O GEMV + Gate/Up GEMV ×2 + SwiGLU + Down GEMV + 2 residual adds)
```

---

## Kernel shape table

### Prefill path

| # | Kernel | Shape / tile config | Tiles | Cosine | max_abs / max_rel | Profile | Status |
|---|---|---|---|---|---|---|---|
| 1 | RMSNorm 2D (attn_norm + ffn_norm + final_norm) | M=2048, N=576, herd_x=8, vector_size=16 | 8×1 | TBD | TBD | TBD | pending |
| 2 | GEMM Q/O proj | M=2048, K=576, N=576 | 8×4 | TBD | TBD | TBD | pending |
| 3 | GEMM K/V proj | M=2048, K=576, N=192 | 8×4 | TBD | TBD | TBD | pending |
| 4 | GEMM Gate/Up proj | M=2048, K=576, N=1536 | 8×4 | TBD | TBD | TBD | pending |
| 5 | GEMM Down proj | M=2048, K=1536, N=576 | 8×4 | TBD | TBD | TBD | pending |
| 6 | RoPE 2D Q | outer 2048×576, hd=64, herd_x=8 | 8×1 | TBD | TBD | TBD | pending |
| 7 | RoPE 2D K | outer 2048×192, hd=64, herd_x=8 | 8×1 | TBD | TBD | TBD | pending |
| 8 | **FlashAttention seq-first** | LQ=LK=2048, hd=64, **n_h=9**, n_kv=3 | (2 seg × 4×4 if n_h%2==0) | — | — | — | ❌ **BLOCKED** — see Active blockers |
| 9 | Eltwise Add 2D (post-attn residual) | rows=2048, cols=576 | 8×1 | TBD | TBD | TBD | pending |
| 10 | SwiGLU FFN block (Gate+Up GEMM + SiLU+Mul + Down GEMM) | seq=2048, emb=576, hidden=1536 | 32 (GEMMs) + 8 (SwiGLU) | TBD | TBD | TBD | pending |
| 11 | Eltwise Add 2D (FFN residual) | rows=2048, cols=576 | 8×1 | TBD | TBD | TBD | pending |
| 12 | LM Head GEMV (per partition; assumes 8 partitions like llama3) | M=6144, K=576, tile_m=8 | 8×1 per partition | TBD | TBD | TBD | pending |

### Decode path

| # | Kernel | Shape | Tiles | Cosine | max_abs / max_rel | Profile | Status |
|---|---|---|---|---|---|---|---|
| 13 | RMSNorm 1D (decode) | M=1, N=576 | 1×1 | TBD | TBD | TBD | pending |
| 14 | GEMV Q/O proj | M=576, K=576, tile_m=8, m_input=4 | 8×1 | TBD | TBD | TBD | pending |
| 15 | GEMV K/V proj | M=192, K=576, tile_m=8, m_input=4 | 8×1 (192 / (8×8) = 3 iters) | TBD | TBD | TBD | pending |
| 16 | GEMV Gate/Up proj | M=1536, K=576, tile_m=8, m_input=4 | 8×1 | TBD | TBD | TBD | pending |
| 17 | GEMV Down proj | M=576, K=1536, tile_m=8, m_input=4 | 8×1 | TBD | TBD | TBD | pending |
| 18 | RoPE 1D Q (decode) | n_rows=9, hd=64 | 1×1 | TBD | TBD | TBD | pending |
| 19 | RoPE 1D K (decode) | n_rows=3, hd=64 | 1×1 | TBD | TBD | TBD | pending |
| 20 | Eltwise Add 1D (post-attn + FFN residuals) | n=576, tile_n=72 | 8×1 | TBD | TBD | TBD | pending |

---

## Active blockers

### B1 — FlashAttention seq-first n_heads=9 (ODD) blocked by hardcoded `num_heads_per_unroll=2`

**Symptom (anticipated, not yet run)**: `attn_npu2_seqfirst.py:119` asserts
`num_heads % num_heads_per_unroll == 0` where `num_heads_per_unroll = 2`
is hardcoded local. With n_h=9, this assert fires before compile.

**Same constraint exists in head-first** (`attn_npu2.py`, supported_kernels.md §5).

**Resolution options**:

1. **Make `num_heads_per_unroll` a `build_module()` parameter**, default 2
   (back-compat). For SmolLM2-135M pass `num_heads_per_unroll=1` → 1 segment
   × 4×4 herd = 16 tiles in flight (vs 32) for FA. Loses 50% FA throughput
   but unlocks **arbitrary** odd n_h support across the chain — paper-relevant
   ("system handles models with arbitrary head counts").

   Touches shared infra (`flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`)
   → triggers Phase 7 cross-deployment regression check on llama3 / smollm2_1_7b
   FA path. Default-unchanged risk: low.

2. **Pad n_h 9 → 10** at the host level. q/o projection becomes
   M=640 instead of M=576; pad a zero-head; slice off the padding output.
   Hacky and weakens the "no per-model retuning" paper claim.

3. **Switch SmolLM2-135M to a different first-pick** (e.g., Qwen2.5-0.5B
   has n_h=14 even). Defers the odd-n_h question.

**Recommendation**: Option 1. ~10 LOC change to add the parameter; default
behavior unchanged for existing 6 deployments. Once landed, it permanently
unblocks the family.

---

## Phase 1 progress

(Empty — runs scheduled after B1 resolved or non-FA kernels validated independently.)
