# SmolLM2-135M BF16 on NPU2 — model-specific guide

End-to-end `HuggingFaceTB/SmolLM2-135M` inference on AMD NPU2 (AIE2P).

Reference deployments:
- `../llama3/` (Llama-3.2-1B base) — canonical kernel sequence,
  multi-launch ELF design, KernelCache, BO pre-loading, seq-first layout.
- `../smollm2_1_7b/` (SmolLM2-1.7B sibling) — small vocab (49152), low
  rope_theta, tied embeddings — same pattern applies.
- `../llama32_3b/` (Llama-3.2-3B) — GQA g=3 (same group factor as 135M),
  but with hd=128 instead of hd=64.

Most patterns from the references apply unchanged. This file documents
only what differs.

## Status

Deployment in progress (`deploy-new-llm` skill chain). See
[`TODO.md`](TODO.md) for phase-by-phase status.

## Model config

30 layers, emb_dim=576, n_heads=**9**, head_dim=64, **n_kv_heads=3**
(GQA g=3), ffn_hidden=1536, vocab=49152, BF16, **rope_θ=100000**,
**tied embeddings**, no QKV bias, no Q/K Norm.

## Divergences from siblings

vs `../smollm2_1_7b/` (its closest dir-name sibling):

1. **GQA, not MHA** (`n_kv_heads=3` vs 32): KV cache is 9× smaller per
   layer relative to MHA at the same (n_h, hd). Reuses the standard
   GQA attention path — no MHA degenerate-group handling needed.
2. **Smaller everything**: emb_dim 2048→576, ffn_hidden 8192→1536,
   n_h 32→9. Phase 1 will pick new tile configs for the smaller GEMVs.
3. **n_h=9 (odd, non-power-of-2)**: first deployment with this; previous
   non-pow2 was qwen25 with n_h=12. Watch for FA/attention tile padding
   issues in Phase 1.
4. **Deeper** (30 vs 24 layers): per-layer BO arrays sized to 30; Phase 3
   wires 30 transformer blocks instead of 24.
5. **New RoPE base** (100000 vs 130000): RoPE LUT regenerated.

vs `../llama3/` (the upstream reference):

- hd=64 ✓ same (no Option C / head-first FA needed)
- GQA g=3 (vs g=4 in llama3); reuses GQA-aware attention path
- vocab=49152 (vs 128256); LM-head GEMV needs the smollm2 49k shape (already
  exercised by smollm2_1_7b)
- Tied embeddings; same pattern as smollm2_1_7b

vs `../llama32_3b/` (same GQA g=3):

- 3B uses hd=128 → triggered Option C head-first FA wrapper. 135M uses
  hd=64 → standard seq-first FA path, no wrapper needed.

## Inheritance path

**Inherit** (default path; no kernel-first work):

- `llama3/multi_launch_builder/` fused ELFs (`rms_gemms_rope`, `o_ffn`,
  `rms_gemv_rope`, `o_gemv_ffn`) reused with new shape params.
- `smollm2_1_7b` patterns for tied embeddings + 49k vocab LM-head GEMV.

## File layout convention

This directory contains **only SmolLM2-135M-specific code**. All llama3
orchestration helpers and multi-launch ELF builders are imported from
`../llama3/` at runtime.

SmolLM2-135M-specific code (created by per-phase skills):

- `smollm2_135m_weights.py` — config dataclass + HF loader + RoPE LUT
- `smollm2_135m_reference.py` — CPU F32 reference forward pass
- `smollm2_135m_inference.py` — end-to-end NPU runner (entry point for `make run`)
- `smollm2_135m_phaseN_test.py` — per-phase validation scripts

Imported from `../llama3/` and `../_llm_shared/` (do not copy):

- `llama3_prefill.run_transformer_block`, `preload_prefill_weights`
- `llama3_decode.run_decode_block`, `compile_decode_kernels`
- `llama3_inference._preload_decode_weights`, `_LM_N_PARTITIONS`
- `multi_launch_builder.{rms_gemms_rope,o_ffn,lm_head_gemv,...}`
- `_llm_shared.kernel_builder.external_kernels.compile_all_external_kernels`
- `_llm_shared.KernelCache`, `prepare_air_project`

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Newcomer overview (placeholder until Phase 6) |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design, profile patterns |
| `../smollm2_1_7b/CLAUDE.md` | Closest sibling — small vocab, tied embeddings |
