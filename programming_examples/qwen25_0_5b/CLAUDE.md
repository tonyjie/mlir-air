# Qwen2.5-0.5B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen2.5-0.5B` inference on AMD NPU2 (AIE2P).

Reference deployments:
- `../llama3/` (Llama-3.2-1B base) — canonical kernel sequence,
  multi-launch ELF design, KernelCache, BO pre-loading, seq-first layout.
- `../qwen25_1_5b/` (Qwen2.5-1.5B sibling) — Qwen2-family patterns:
  QKV bias via host-side RoPE-linearity add (`qwen25_bias.py`),
  GQA-aware reindexed padding (`qwen25_pad.py`), tied embeddings,
  rope_θ=1M, vocab=151936.

## Status

Deployment in progress. See [`TODO.md`](TODO.md) for phase-by-phase status.

## Model config

24 layers, emb_dim=**896**, n_heads=**14**, head_dim=**64**, n_kv_heads=2
(GQA g=**7**), ffn_hidden=**4864**, vocab=151936, BF16, rope_θ=1000000,
**tied embeddings**, **QKV bias = True** (Qwen2 default), no Q/K Norm.

`sliding_window=32768` is set in HF config but `use_sliding_window=false`,
so SWA does not engage at our seq_len=2048 (or any seq_len). Safe to deploy
as standard causal attention.

## Divergences from siblings

vs `../qwen25_1_5b/` (closest family sibling):

1. **head_dim=**`64` (vs 1.5B's 128). **No Option C head-first FA wrapper
   needed** — use the standard seq-first FA path (the same path llama3 /
   smollm2_1_7b use). This is a major simplification vs 1.5B.
2. **n_h=14 even** ✓ — FA `num_heads_per_unroll=2` constraint satisfied
   (14 % 2 = 0). Avoids the SmolLM2-135M B1 blocker.
3. **GQA group = 7** (vs 1.5B's 6). Both are odd / non-power-of-2 →
   reuses the GQA-aware reindexed padding path from `qwen25_pad.py`.
4. **Smaller everything**: emb 1536→896, ffn 8960→4864, n_h 12→14
   (slightly more), 28→24 layers.
5. **K=4864 < 8160** ✓ — no `down_k_split` needed (Rule B not engaged
   at this size).

vs `../llama3/`:

- hd=64 ✓ (no head-first wrapper)
- vocab=151936 (vs 128256); LM-head GEMV needs the qwen25 152k shape
  (already exercised by qwen25_1_5b)
- QKV bias is new vs llama3 — handled via Qwen25 host-side bias add
- Tied embeddings (llama3 also tied)

## Inheritance path

**Inherit** (default path; no kernel-first work):

- `llama3/multi_launch_builder/` fused ELFs reused with new shape params.
- `qwen25_1_5b` helpers reused as-is or imported via sys.path:
  - `qwen25_bias.py` — host-side bias add (RoPE linearity trick)
  - `qwen25_pad.py` — GQA-aware reindexed padding for non-aligned dims

Phase 1 will verify whether qwen25_1_5b helpers work unchanged at the 0.5B
shapes (emb=896, ffn=4864) or need new tile config selections.

## File layout convention

Minimal scaffold + sys.path imports. Qwen2.5-0.5B-specific code
(created by per-phase skills):

- `qwen25_0_5b_weights.py` — config dataclass + HF safetensors loader (incl. QKV bias)
- `qwen25_0_5b_reference.py` — CPU F32 reference forward pass
- `qwen25_0_5b_inference.py` — end-to-end NPU runner (entry point for `make run`)
- `qwen25_0_5b_phaseN_test.py` — per-phase validation scripts

Imported from `../llama3/`, `../qwen25_1_5b/`, `../_llm_shared/` (do not copy):

- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}`
- `qwen25_bias.apply_qkv_bias_postnorm` (or whichever entry point)
- `qwen25_pad.*` for GQA-aware padding helpers
- `_llm_shared.kernel_builder.external_kernels.compile_all_external_kernels`
- `_llm_shared.KernelCache`, `prepare_air_project`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../qwen25_1_5b/CLAUDE.md` | Closest sibling — Qwen2 family conventions |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design |
