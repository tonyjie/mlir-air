# SmolLM2-1.7B BF16 on NPU2 — model-specific guide

End-to-end `HuggingFaceTB/SmolLM2-1.7B` inference on AMD NPU2 (AIE2P).
Reference deployment: `../llama3/` (Llama-3.2-1B base) — see its
`CLAUDE.md` for the canonical kernel sequence, multi-launch ELF design,
KernelCache pattern, BO pre-loading, and seq-first layout. Most of those
patterns apply unchanged here; this file documents only what *differs*.

## Status

**Deployed and operational** (2026-04-17). All 7 phases of `deploy-new-llm`
passed; end-to-end NPU inference (prefill + decode) is wired up via
`smollm2_inference.py` (entry point for `make run`).

Performance (validated):
- **Prefill**: 2.25 s for 24 layers + KV-cache extraction (94 ms/layer);
  standalone Phase-4 prefill (no extraction) is 1.88 s / 79 ms-per-layer
- **Decode**: 137 ms/token (7.3 tok/s); per-layer rate at parity with llama3
- See `docs/development_progress/phase6_finalize.md` for the perf summary.

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

Inherited `llama3_*.py` filenames are kept per Lesson 1 of the smoke test
(rename is high-friction; the existing config-driven helpers worked unchanged
for SmolLM2). SmolLM2-specific code:

- `smollm2_weights.py` — config dataclass + HF safetensors loader + RoPE LUT
- `smollm2_reference.py` — CPU F32 reference forward pass
- `smollm2_inference.py` — **end-to-end NPU runner** (prefill with K/V extraction
  + first-token NPU LM Head + decode loop). Entry point for `make run`.
- `smollm2_phaseN_test.py` — per-phase validation scripts (used during deploy);
  individually invokable via `make run-block` / `run-full` / `run-prefill` /
  `run-decode-only` / `run-reference`.

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Newcomer overview, perf, usage, arch comparison |
| [TODO.md](TODO.md) | Phase status and active blockers |
| [docs/development_progress/progress.md](docs/development_progress/progress.md) | Phase log |
| [docs/development_progress/phase6_finalize.md](docs/development_progress/phase6_finalize.md) | End-to-end perf summary + reusable-pattern audit |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | Novel failures + root-cause fixes |
| [docs/development_progress/debug_log.md](docs/development_progress/debug_log.md) | Debug-recipe firings |
| `../llama3/CLAUDE.md` | Canonical kernel sequence, multi-launch design, profile patterns |
