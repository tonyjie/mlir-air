# Qwen2.5-3B BF16 on NPU2 — model-specific guide

End-to-end `Qwen/Qwen2.5-3B` inference on AMD NPU2 (AIE2P).

Reference deployments:
- `../llama3/` (Llama-3.2-1B base) — canonical kernel sequence,
  multi-launch ELF design, KernelCache, BO pre-loading.
- `../qwen25_1_5b/` (Qwen2.5-1.5B sibling) — closest match: same
  Qwen2 family with QKV bias, GQA-aware reindexed padding, tied
  embeddings, rope_θ=1M, vocab=151936, **head_dim=128 → uses Option C
  head-first FA wrapper**.
- `../llama32_3b/` (Llama-3.2-3B) — head_dim=128 + 28-layer reference
  (we go even deeper at 36 layers).

## Status

Deployment in progress. See [`TODO.md`](TODO.md) for phase status,
[`docs/development_progress/phase_timing.md`](docs/development_progress/phase_timing.md)
for per-phase wall-clock.

## Model config

36 layers, emb_dim=**2048** (1024-aligned ✓ — no emb padding needed),
n_heads=16, head_dim=128, n_kv_heads=2 (GQA g=**8**), ffn_hidden=11008
(NOT 1024-aligned; pad → 11264 = 11×1024), vocab=151936, BF16,
rope_θ=1000000, **tied embeddings**, **QKV bias = True**.

`sliding_window=32768` set in HF config but `use_sliding_window=false`
→ SWA disabled, safe to deploy as standard causal attention.

## Divergences from siblings

vs `../qwen25_1_5b/` (closest sibling):

1. **Deeper**: 36 layers (vs 28) — deepest deployment in the catalog.
2. **emb_dim=2048 NOT padded** (1.5B padded 1536→2048). 2048 is already
   1024-aligned. Saves ~33% compute on Q/O GEMMs vs the padded 1.5B path.
3. **GQA group=8** (vs 1.5B's 6). Same group as our padded qwen25_0_5b.
   W1 (NPU seq-first FA precision drop at GQA-imbalanced shapes) does
   NOT apply here because hd=128 routes through Option C head-first FA
   path (precision-clean). W1 is seq-first-FA-specific.
4. **n_h=16** (vs 12) — bigger but still even ✓ FA OK.
5. **hidden=11008** (vs 8960). Pad to **12288** (12×1024) for prefill —
   the 11264 (11×1024) variant compiled but runtime-hung at seq=2048
   (BD pool tightness even though 1024-aligned). K-split for Down GEMV
   in decode: `down_k_split=86` (86×128=11008, satisfies Rule B/D).
6. Otherwise: head_dim=128 → same Option C head-first wrapper, same
   QKV bias add via RoPE linearity, same 10×16384 LM head partition.

## Inheritance path

**Inherit** (default; same path as qwen25_1_5b):
- `llama3/multi_launch_builder/` fused ELFs reused with new shape.
- `qwen25_1_5b` helpers (`qwen25_pad.py` + `qwen25_bias.py`) imported
  via sys.path. Both are model-agnostic.
- New: `qwen25_3b_decode_setup.py` for `mv_k11264.o` (DIM_M_OUTPUT=2)
  + `down_k_split=88`.
- Option C wrapper from `_llm_shared/phase_helpers/headfirst_fa.py`
  (head_dim=128 path).

## Watch list

- **W1 (NPU FA precision drop at GQA g=8 + n_kv=2)** — qwen25_0_5b hit
  this; qwen25_3b shares the exact same GQA shape. If it reproduces,
  this is the **second qualifying deployment** for `skill-update.md` B2.
- **Down K-split factor** is new at K=11264 (vs 1.5B's K=8960 with
  split=70). Need `down_k_split=88` (88×128=11264). Verify Rule B/C/D.
- **Compile time**: 36 layers × multi-launch ELFs may take longer than
  the 28-layer 1.5B compile.

## File layout convention

Minimal scaffold + sys.path imports. Qwen2.5-3B-specific code
(produced by per-phase skills):

- `qwen25_3b_weights.py` — config + HF safetensors loader (incl. QKV bias)
- `qwen25_3b_reference.py` — CPU F32 reference forward pass
- `qwen25_3b_inference.py` — end-to-end NPU runner (entry for `make run`)
- `qwen25_3b_phaseN_test.py` — per-phase validation scripts
- `qwen25_3b_decode_setup.py` — `mv_k11264.o` + 10-partition LM head GEMV

Imported from `../llama3/`, `../qwen25_1_5b/`, `../_llm_shared/`:

- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}`
- `qwen25_bias.*` (host post-RoPE bias add)
- `qwen25_pad.*` (GQA-aware padding for hidden_dim 11008→11264)
- `_llm_shared.kernel_builder.external_kernels.compile_all_external_kernels`
- `_llm_shared.phase_helpers.headfirst_fa.*` (Option C, hd=128)
- `_llm_shared.KernelCache`, `prepare_air_project`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status, active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/phase_timing.md](docs/development_progress/phase_timing.md) | Per-phase wall-clock |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../qwen25_1_5b/CLAUDE.md` | Closest sibling — Qwen2 family conventions |
| `../qwen25_0_5b/CLAUDE.md` | Same GQA shape (g=8) for W1 reference |
| `../llama32_3b/CLAUDE.md` | head_dim=128 + Option C wrapper reference |
