# SmolLM2-1.7B BF16 on NPU2 — model-specific guide

End-to-end `HuggingFaceTB/SmolLM2-1.7B` inference on AMD NPU2 (AIE2P).
Reference deployment: `../llama3/` (Llama-3.2-1B base) — see its
`CLAUDE.md` for the canonical kernel sequence, multi-launch ELF design,
KernelCache pattern, BO pre-loading, and seq-first layout. Most of those
patterns apply unchanged here; this file documents only what *differs*.

## Status

Bootstrapping. See `TODO.md` and `docs/development_progress/progress.md`.

## Model config

24 layers, emb_dim=2048, n_heads=32, head_dim=64, **n_kv_heads=32 (MHA)**,
hidden_dim=8192, **vocab=49152**, BF16, **rope_θ=130000**, **tied embeddings**.

## Divergences from `llama3/`

1. **MHA, not GQA** (`n_kv_heads=32`): KV cache is 4× larger. Most attention
   kernels accept this as degenerate GQA (group_size=1) — verify in Phase 1.
2. **Tied embeddings**: `lm_head.weight` is `embed_tokens.weight`. Weight
   loader (`smollm2_weights.py`) special-cases this; LM-head GEMV reads the
   embedding table directly.
3. **Smaller vocab** (49152 vs 128256): LM-head GEMV needs a new shape.
   Phase 1 must add it to the kernel sweep.
4. **New RoPE base** (130000 vs 500000): RoPE LUT regenerated from this base.
5. **Deeper** (24 vs 16 layers): per-layer BO arrays sized to 24; Phase 3
   wires 24 transformer blocks instead of 16.

## File layout convention

Inherited `llama3_*.py` filenames are kept temporarily per Lesson 1 of the
smoke test (rename is high-friction; defer until Phase 0 confirms the actual
divergence depth in code paths). Phase 0 produces:
- `smollm2_weights.py` — config dataclass, HF weight loader, RoPE LUT
- `smollm2_reference.py` — CPU F32 reference

If the inference / prefill / decode entry points need real divergence (e.g.,
to handle MHA layout differently), Phase 0 will either parameterize the
existing files or fork them — that decision is logged in this file's
"Divergences" section above.

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Newcomer overview + arch comparison table |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design, profile patterns |
