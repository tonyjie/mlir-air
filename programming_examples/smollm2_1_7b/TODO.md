# Deployment: smollm2_1_7b

Deployment of `HuggingFaceTB/SmolLM2-1.7B` on NPU2 via the `deploy-new-llm`
skill chain. Architecturally this is a structural twin of Llama-3.2-1B
(same RMSNorm + SwiGLU + RoPE half-split + emb_dim/head_dim/hidden), with
four focused divergences:

1. **Depth**: 24 layers (vs 16) — Phase 3 wires more blocks
2. **MHA, not GQA**: `n_kv_heads=32` (degenerate GQA from the kernels' POV);
   KV cache is 4× larger than Llama-3.2-1B
3. **Tied embeddings**: `lm_head.weight = embed_tokens.weight` — weight loader
   special-case
4. **Smaller vocab + new RoPE base**: vocab=49152, rope_θ=130000 — LM-head
   GEMV needs a new shape; RoPE LUT must be regenerated

Reference deployment: `programming_examples/llama3/`. Inherited scaffold
keeps `llama3_*.py` filenames per Lesson 1 of the smoke test (rename
deferred until Phase 0 confirms divergence depth).

## Phase status
- [x] 0: Bootstrap (PASSED 2026-04-17 — top-1 " Paris", HF logits corr=0.99999978)
- [x] 1: Per-kernel shapes (PASSED 2026-04-17 — 16/16 kernels: 12 drop-in, 4 parametric-recompile; LM Head partition scheme flagged for Phase 4/5)
- [x] 2: Single block (PASSED 2026-04-17 — cosine 0.999 per-position; MAE 0.025 matches BF16-production baseline; NPU FA with MHA validated)
- [x] 3: Full model (PASSED 2026-04-17 — 3/3 top-1 matches: " Paris", " ", " blue"; per-layer cos min=0.974 > 0.95; 24-layer NPU prefill ~2.0s)
- [x] 4: Prefill perf (PASSED 2026-04-17 — warm prefill 1.88s NPU / 2.41s wall; per-layer 79ms = parity with llama3 despite 4× MHA compute; 4/5 patterns applied)
- [x] 5: Decode perf (PASSED 2026-04-17 — 136.4 ms/token, 7.3 tok/s; 3/3 NPU/CPU match; "The capital of France is Paris.\n\nThe capital of France"; 5/5 patterns; per-layer rate at parity with llama3)
- [x] 6: Finalize (PASSED 2026-04-17 — see docs/development_progress/phase6_finalize.md; survey corrected; deployment complete)

## Active blockers
(none yet)

## Resolved config (pulled from HF `config.json`)
```
n_layers:     24
emb_dim:      2048
n_heads:      32
n_kv_heads:   32       # MHA — handled as degenerate GQA (group_size=1)
head_dim:     64
hidden_dim:   8192
vocab_size:   49152
rope_theta:   130000.0  # NB: edge-llm-candidates.md lists 10k — survey is stale
rms_norm_eps: 1e-5
max_pos:      8192
tie_word_embeddings: true
torch_dtype:  bfloat16
```
