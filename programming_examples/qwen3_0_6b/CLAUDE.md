# Qwen3-0.6B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen3-0.6B` inference on AMD NPU2 (AIE2P).
Reference deployments:
- `../llama3/` — canonical kernel sequence, multi-launch ELF, KernelCache, BO pre-loading, seq-first layout
- `../llama32_3b/` — head_dim=128 + Option C head-first FA wrapper (this model also requires it)
- `../qwen25_1_5b/` — Qwen-family weights, GQA reindexing for non-power-of-2 group sizes

## Status

**Production-ready** (2026-04-20). End-to-end full-NPU inference operational
via `qwen3_inference.py` (`make run`). Phase A (Q/K Norm on NPU per-leaf),
Phase B (3 fused decode ELFs), and host-side optimizations (pre-transposed
weights, per-layer arg cache, BO preload) are all landed.

Performance (validated):
- **NPU Prefill**: 2.09 s warm @ seq_len=2048 (74.8 ms/layer × 28 layers)
- **NPU Decode**: 0.09 s/token (10.7 tok/s) — at parity with llama3-1B
- 3 decode ELFs: `rms_attn_gemvs_qknorm_rope` (8 launches, fuses RMSNorm +
  Q/K/V GEMV + Q/K Norm + RoPE Q/K), `o_gemv_ffn_silu` (8 launches with 3-K
  matvec extern rename), `lm_head_gemv` (10 partitions × 16384)

Functional gates verified:
- 6/6 canonical prompts pass dynamic decisive/competitive gate (NPU prefill + NPU LM head)
- 6/6 NPU decode tokens within CPU top-5 (multi-token verify; 5/6 exact top-1, 1 BF16-reorder soft-pass)
- NPU decode 10.8× faster than CPU decode

See [docs/development_progress/phase_b_fusion.md](docs/development_progress/phase_b_fusion.md)
for the fusion design and 10× host-side speedup details.
See [LESSONS.md](docs/development_progress/LESSONS.md) for the lessons
captured during this deployment.

## Model config

28 layers, emb_dim=1024, n_heads=16, **head_dim=128**, n_kv_heads=8 (GQA group=2),
hidden_dim=3072, vocab=151936, BF16, **rope_θ=1,000,000**, **tied embeddings**
(but lm_head.weight is also stored explicitly in safetensors), **NO QKV bias**,
**NEW Q/K Norm** (per-layer per-head RMSNorm BEFORE RoPE).

## Divergences from prior deployments

1. **NEW Q/K Norm** (Qwen3-only, biggest divergence): per-layer
   `self_attn.q_norm.weight` and `self_attn.k_norm.weight`, each `(head_dim,)`.
   Applied as per-head RMSNorm AFTER Q/K projection but BEFORE RoPE.
   - Shared host helper: `_llm_shared/phase_helpers/qk_norm.py`
   - Cannot fuse into existing rms_gemms_rope ELF via linearity trick
     (RMSNorm doesn't commute with RoPE for asymmetric weights).
   - **Two integration paths:**
     a. **Split-ELF approach**: use predecessor `rms_attn_gemms_multi.py` +
        host Q/K Norm + `rope_qk_multi.py`. Slower (3 XRT calls instead of 1)
        but minimal new kernel work.
     b. **New on-tile RMSNorm-per-head kernel**: fuse into rms_gemms_rope.
        Faster but requires new C kernel + ELF integration. **Defer this.**
2. **NO QKV bias** (vs Qwen2.5): unlike Qwen2.5, Qwen3 dense uses
   `attention_bias=False`. Skip the bias-add wrapper used by qwen25_1_5b.
3. **head_dim=128** (same as `llama32_3b/`, `qwen25_1_5b/`): use Option C
   head-first FA wrapper from `_llm_shared/phase_helpers/headfirst_fa.py` —
   seq-first FA is broken for `dk_chunks > 1`.
4. **rope_θ = 1,000,000** (same as Qwen2.5): RoPE LUT regenerated.
5. **GQA group=2** (vs Qwen2.5 group=6, llama32_3b group=3): minimal sharing
   per KV head. Group size IS a power of 2 → no GQA-reindex padding needed
   (unlike Qwen2.5 which needed phantom-Q-head padding for emb_dim 1536→2048).
6. **BD-friendly shapes**: emb_dim=1024 and hidden_dim=3072 are clean multiples
   of 1024 — no padding gymnastics expected at GEMM tile config selection.
7. **Smaller** (28 layers but only emb_dim=1024) — much smaller than Llama-3.2-3B
   (emb_dim=3072) at same depth. Memory footprint should be ~1.2 GB BF16 weights.
8. **Tied embeddings stored explicitly**: Qwen3-0.6B safetensors carry
   `lm_head.weight` even though `tie_word_embeddings=True` in config. Loader
   uses the explicit lm_head when present.

## File layout convention

Minimal scaffold + sys.path imports (recommended pattern; see
`deploy-new-llm` skill). This directory contains **only Qwen3-specific code**;
all orchestration helpers and multi-launch ELF builders are imported from
`../llama3/` and `../_llm_shared/` at runtime.

Qwen3-specific code:
- `qwen3_weights.py` — config dataclass + HF safetensors loader (Q/K Norm + no QKV bias) + RoPE LUT
- `qwen3_reference.py` — CPU F32 reference forward pass (Q/K Norm BEFORE RoPE)
- `qwen3_phaseN_test.py` — per-phase validation scripts
- `qwen3_inference.py` — end-to-end NPU runner

Imported from `../llama3/` (do not copy):
- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}` (current production)
- `multi_launch_builder.superseded.{rms_attn_gemms_multi,rope_qk_multi}` (split-ELF path for Q/K Norm)

Imported from `../_llm_shared/`:
- `KernelCache`, `prepare_air_project`
- `compile_all_external_kernels` (rope_halfsplit, silu_and_mul, attn_npu2, mv)
- `compile_attn_npu2_split` (per-tile NPU FA for head_dim=128)
- `phase_helpers.headfirst_fa.patch_run_cached_for_headfirst_fa` (Option C wrapper)
- `phase_helpers.qk_norm.apply_qk_norm` (NEW host wrapper for Q/K Norm)
- `phase_helpers.{metrics, canonical_prompts, decode_setup, ...}`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Newcomer overview (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design |
| `../qwen25_1_5b/CLAUDE.md` | Qwen-family precedent (QKV bias version) |
| `../llama32_3b/CLAUDE.md` | head_dim=128 + Option C FA wrapper reference |
