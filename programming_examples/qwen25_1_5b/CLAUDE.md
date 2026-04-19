# Qwen2.5-1.5B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen2.5-1.5B` inference on AMD NPU2 (AIE2P).
Reference deployments: `../llama3/` (Llama-3.2-1B base) for the canonical
kernel sequence, multi-launch ELF design, KernelCache, BO pre-loading;
`../llama32_3b/` for the **head_dim=128 + Option C head-first FA wrapper**
that this model also requires.

## Status

**Deployed and operational** (2026-04-19). All 7 phases of `deploy-new-llm`
PASSED. End-to-end NPU inference (prefill + decode) wired up via
`qwen25_inference.py` (entry point for `make run`).

Performance (validated):
- **Prefill**: 2.4 s for 28 layers (85 ms/layer, seq_len=2048) via NPU FA
  Option C head-first wrapper at head_dim=128
- **Decode**: 216 ms/token (4.6 tok/s); per-layer rate matches llama32_3b
- See `docs/development_progress/phase6_finalize.md` for the full perf summary.

## Model config

28 layers, emb_dim=1536, n_heads=12, **head_dim=128**, n_kv_heads=2 (GQA group=6),
hidden_dim=8960, vocab=151936, BF16, **rope_θ=1,000,000**, **tied embeddings**,
**QKV bias = True**.

## Divergences from prior deployments

1. **QKV bias** (NEW): `q_proj.bias`, `k_proj.bias`, `v_proj.bias` exist
   (1536 / 256 / 256 floats each). Existing `rms_gemms_rope` builder produces
   bias-free GEMMs — needs an eltwise-add pass after each Q/K/V GEMM, or a
   bias-aware kernel variant. Plan in Phase 1.
2. **head_dim=128** (same as `llama32_3b/`): use Option C head-first FA wrapper
   from `_llm_shared/phase_helpers/headfirst_fa.py` — seq-first FA is broken
   for `dk_chunks > 1`.
3. **rope_θ = 1,000,000** (vs Llama 500k / SmolLM2 130k): RoPE LUT regenerated.
4. **Larger vocab** (151936 vs Llama 128256 / SmolLM2 49152): LM-head GEMV
   needs new shape. Phase 1 must add it to the kernel sweep.
5. **Tied embeddings** (like SmolLM2): `lm_head.weight === embed_tokens.weight`.
   Weight loader special-cases this; LM-head GEMV reads embedding table.
6. **GQA group=6** (vs Llama-3.2-3B group=3): more sharing per KV head.
7. **Deeper than 1B** (28 layers, like llama32_3b): per-layer BO arrays sized
   to 28; Phase 3 wires 28 transformer blocks.

## File layout convention

Minimal scaffold + sys.path imports (recommended pattern; see
`deploy-new-llm` skill). This directory contains **only Qwen2.5-specific
code**; all orchestration helpers and multi-launch ELF builders are imported
from `../llama3/` and `../_llm_shared/` at runtime.

Qwen2.5-specific code (added by phase skills):

- `qwen25_weights.py` — config dataclass + HF safetensors loader (incl. QKV bias)
- `qwen25_reference.py` — CPU F32 reference forward pass
- `qwen25_inference.py` — end-to-end NPU runner
- `qwen25_phaseN_test.py` — per-phase validation scripts

Imported from `../llama3/` (do not copy):
- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}`

Imported from `../_llm_shared/`:
- `KernelCache`, `prepare_air_project`
- `compile_all_external_kernels` (rope_halfsplit, silu_and_mul, attn_npu2, mv)
- `compile_attn_npu2_split` (per-tile NPU FA for head_dim=128)
- `phase_helpers.headfirst_fa.patch_run_cached_for_headfirst_fa` (Option C wrapper)
- `phase_helpers.{metrics, canonical_prompts, decode_setup, ...}`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design |
| `../llama32_3b/CLAUDE.md` | head_dim=128 + Option C FA wrapper reference |
