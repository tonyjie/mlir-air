# Llama-3.2-3B BF16 on NPU2 — model-specific guide

End-to-end `meta-llama/Llama-3.2-3B` inference on AMD NPU2 (AIE2P).
Reference deployment: `../llama3/` (Llama-3.2-1B base) — see its
`CLAUDE.md` for the canonical kernel sequence, multi-launch ELF design,
KernelCache pattern, BO pre-loading, and seq-first layout. Most of those
patterns apply unchanged here; this file documents only what *differs*.

## Status

**Scaffolded** (2026-04-17) — Phase 0 not yet started. See
`TODO.md` for live phase status.

Per Path A roadmap (`docs/superpowers/edge-llm-candidates.md`),
Llama-3.2-3B is the second Tier-A deployment after smollm2_1_7b.

## Model config

28 layers, emb_dim=3072, n_heads=24, **head_dim=128**, n_kv_heads=8 (GQA
group=3), hidden_dim=8192, vocab=128256, BF16, **rope_θ=500000**,
tied embeddings, max_pos=131072 with rope_scaling (llama3 long-context).

## Divergences from `llama3/` (Llama-3.2-1B)

1. **head_dim=128, not 64** — biggest known blocker. RoPE half-split kernel
   (`_llm_shared/kernel_builder/rope_halfsplit.cc`) and FlashAttention kernel
   (`flash_attention/kernel_fusion_based/attn_npu2.cc`) hardcode head_dim=64.
   Phase 1 will surface this; expect 4–8 hours of manual `.cc` work.
2. **Deeper** (28 vs 16 layers) — per-layer BO arrays sized to 28; Phase 3
   wires 28 transformer blocks instead of 16.
3. **Wider** (emb_dim=3072 vs 2048) — divisible by 24 heads × 128 head_dim.
   GEMM tile configs may need re-tuning.
4. **GQA group=3** (24 heads / 8 KV-heads). KV cache size sits between
   Llama-3.2-1B (group=4) and SmolLM2 (MHA, group=1).
5. **rope_scaling=llama3** with `factor=32`, `low_freq_factor=1.0`,
   `high_freq_factor=4.0`, `original_max_position_embeddings=8192`. Inert for
   `seq_len <= 8192`. Defer the long-context wavelength remap to a Phase 6
   follow-up unless tested with `seq_len > 8192`.
6. **Memory footprint** — model is ~6 GB BF16 weights; with kernel BOs and
   K/V cache the runtime sits ~11 GB on NPU2's 16 GB DRAM. Watch BO budget
   in Phase 4/5.

## File layout convention

This directory contains **only Llama-3.2-3B-specific code**. All llama3
orchestration helpers and multi-launch ELF builders are imported from
`../llama3/` at runtime via the sys.path bootstrap block at the top of
each script. This is the recommended pattern for Tier-A model deployments
(see `deploy-new-llm` skill).

Llama-3.2-3B-specific code (per-phase skills produce these):
- `llama32_3b_weights.py` — config dataclass + HF safetensors loader + RoPE LUT
- `llama32_3b_reference.py` — CPU F32 reference forward pass
- `llama32_3b_phaseN_test.py` — per-phase validation scripts
- `llama32_3b_inference.py` — end-to-end NPU runner (Phase 6)

Imported from `../llama3/` (do not copy):
- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`, `_LM_GEMV_BACKEND`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}`

Imported from `../_llm_shared/`:
- `KernelCache`, `prepare_air_project`
- `compile_all_external_kernels` (silu_and_mul, rope_halfsplit, attn_npu2, mv)

## Documentation

| Doc | Content |
|-----|---------|
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design, profile patterns |
| `../smollm2_1_7b/CLAUDE.md` | Tier-A precedent (different divergences: MHA, vocab, RoPE base) |
