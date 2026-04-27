# Deployment: smollm2_135m  (HuggingFaceTB/SmolLM2-135M)

## Status: ⏸ PAUSED (FA odd-n_h blocker, see Active blockers)

## Phase status
- [x] 0: Build CPU Oracle  (PASS 2026-04-27 — per-layer cos min 0.999996, final cos 0.99999976, top-1 match)
- [⏸] 1: Kernel Validation  (paused — see Active blockers B1)
- [ ] 2: Single-Block Validation
- [ ] 3: Full-Model Validation
- [ ] 4: Prefill Optimization
- [ ] 5: Decode Optimization
- [ ] 6: Finalize & Learn
- [ ] 7: Independent Evaluation

## Active blockers

### B1 (2026-04-27) — FA `num_heads_per_unroll=2` hardcoded blocks n_h=9

**Where**: `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:118-122`
(and same pattern in head-first `attn_npu2.py`).

**Symptom**: SmolLM2-135M has n_h=**9** (odd, non-power-of-2). FA seq-first
asserts `num_heads % 2 == 0` at compile time → assert fires before NPU run.

**Root cause**: `num_heads_per_unroll = 2` is a hardcoded local in
`build_module()`. Drives the segment-unroll factor. All currently-deployed
models have even n_h (32, 24, 32, 12, 16, 16) so this never came up.

**Resolution path** (deferred — see model_support_list.md "Model status board"):
make `num_heads_per_unroll` a `build_module()` parameter, default 2 (back-compat).
For SmolLM2-135M pass `num_heads_per_unroll=1` → 1 segment × 4×4 herd = 16 tiles
(vs 32). Touches shared infra, triggers Phase 7 cross-deployment regression.

**To resume this deployment**: land Option 1 fix → re-run
`Skill: deploy-new-llm HuggingFaceTB/SmolLM2-135M` from Phase 1.

Phase 0 (CPU oracle) is already PASS-verified; resume can skip it.

## Resolved config (pulled from HF — `HuggingFaceTB/SmolLM2-135M`)

| Field | Value |
|---|---|
| n_layers              | 30 |
| emb_dim (hidden_size) | 576 |
| n_heads               | 9 (odd, non-power-of-2 — first deployment with this) |
| n_kv_heads            | 3 → GQA group g=3 (same factor as llama32_3b) |
| head_dim              | 64 |
| ffn_hidden            | 1536 |
| vocab                 | 49152 (same as smollm2_1_7b) |
| rope_theta            | 100000 (new — we have 130k, 500k, 1M; LUT regen only) |
| tie_word_embeddings   | true |
| attention_bias        | false (no QKV bias) |
| qk_norm               | none |
| dtype                 | bf16 |

## Inheritance path

**inherit** (default): reuse `llama3/multi_launch_builder/` fused ELFs +
`smollm2_1_7b` patterns for tied embeddings + 49k vocab LM-head GEMV.

## Watch list

- **n_h=9 (odd, non-pow-of-2)**: first deployment with odd head count.
  Phase 1 attention tile padding may need a new selection.
- **Tiny dims** (emb 576, ffn 1536): may stress the lower bound of tile
  configs that target hidden ≥ 2048.
- **Per-layer BO arrays sized to 30 layers** (deepest deployment so far).
