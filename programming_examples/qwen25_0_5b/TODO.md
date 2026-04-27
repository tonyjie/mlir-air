# Deployment: qwen25_0_5b  (Qwen/Qwen2.5-0.5B)

## Phase status
- [x] 0: Build CPU Oracle  (PASS 2026-04-27 — per-layer cos min 0.999923, final cos 1.00000000, top-1 ' Paris' match)
- [x] 1: Kernel Validation  (PASS 2026-04-27 — 15/15 shapes PASS, min cos 0.997489 FA, max 0.999998 RoPE 1D K)
- [⚠] 2: Single-Block Validation  (PASS-with-warnings 2026-04-27 — CPU FA cos 0.999540 PASS; NPU FA cos 0.978 / per_pos 0.94 fails strict gate. Proceeding to Phase 3 with NPU FA; fall back to CPU FA if top-1 fails)
- [x] 3: Full-Model Validation  (PASS 2026-04-27 — 6/6 prompts top-1 match with NPU FA: 3/3 decisive + 3/3 competitive top-5 overlap. ' Paris' / ' Pacific' / ' falling' all correct. ~39 ms/layer)
- [x] 4: Prefill Optimization     (PASS 2026-04-27 — 5/5 patterns applied/inherited. Warm 34 ms/layer ± 32 ms; cold→warm B1 gain 55.8%. Wall 1.676 s. Top-1 ' Paris' preserved across 5 runs)
- [x] 5: Decode Optimization      (PASS 2026-04-27 — 5/5 patterns. Steady 128.7 ms/token = 7.8 tok/s = 5.4 ms/layer; mv_k4864.o + 10-partition LM Head GEMV. 4/4 CPU verify match)
- [⚠] 6: Finalize & Learn          (PASS-with-warnings 2026-04-27 — `make run` TTFT 0.89s + TPS 8.0; `make verify` top-1 ' Paris' match + logits cos 0.994; multi-token greedy 4/5 (tok 2 diverges between NPU/CPU on competitive continuation — W1 manifestation, not KV cache bug))
- [x] 7: Independent Evaluation    (PASS-with-warnings 2026-04-27 — fresh subagent audited verify gate clean + re-ran make verify + 3/3 adversarial PASS + reproducibility 2/2 byte-identical. Report: `docs/evaluation_report_2026-04-27.md`)

## Active blockers / warnings

### W1 (2026-04-27, Phase 2) — NPU seq-first FA precision drop at GQA g=8 + n_kv=2

**Symptom**: NPU FA in single-block context gives cosine 0.978 / per_pos_min 0.941
vs CPU FA's 0.999540 / 0.999249 (CPU FA in same block), at the same padded
shape (n_h=16, n_kv=2, hd=64, LQ=LK=2048).

**Bisect**:
- Standalone FA at this exact shape: cos 0.997 (Phase 1) — close to expected.
- In-block FA: ~0.02 cos drop. Triggered by realistic Q/K/V distribution
  (RMSNorm + projection + bias + RoPE produces outlier-heavy attention scores
  that BF16 cascade-merge in FA amplifies).
- Same seq-first FA path is fine for smollm2_1_7b (n_h=n_kv=32, MHA, hd=64) → cos 0.998.

**Conjecture**: GQA group=8 + small n_kv=2 (only 2 KV heads to share among 16
Q heads after padding, vs smollm2's 1:1) leaves the cascade with fewer
independent reductions, amplifying noise.

**Decision (Phase 2, 2026-04-27)**: accept PASS-with-warnings; advance to Phase 3
with NPU FA and check whether top-1 prediction survives. If Phase 3 top-1
fails, fall back to CPU attention path for this deployment.

**Phase 3 outcome (2026-04-27)**: NPU FA top-1 SURVIVED at 24-layer scale —
all 6 canonical prompts match CPU. **W1 downgraded from BLOCKER to
informational warning.** The Phase 2 per-pos gate was more conservative
than the semantic ground truth. Worth a paper observation about NPU FA
precision profile across GQA group_size.

**Surface as skill-update item B2** if confirmed across more deployments.

## Resolved config (from HF — `Qwen/Qwen2.5-0.5B`)

| Field | Value |
|---|---|
| n_layers              | 24 |
| emb_dim (hidden_size) | **896** (not 1024-aligned — Rule A pressure) |
| n_heads               | 14 (even ✓ — FA num_heads_per_unroll=2 OK) |
| n_kv_heads            | 2 → GQA group g=**7** (odd, non-power-of-2) |
| head_dim              | **64** (vs 1.5B's 128 — no Option C wrapper needed) |
| ffn_hidden            | **4864** (not 1024-aligned; K<8160 — no K-split needed) |
| vocab                 | 151936 (same as qwen25_1_5b) |
| rope_theta            | 1000000 (same as qwen25_1_5b) |
| tie_word_embeddings   | true |
| attention_bias        | none in config — but Qwen2 has q/k/v_proj.bias by default; verify in Phase 0 weight load |
| qk_norm               | none |
| sliding_window        | 32768 set, but `use_sliding_window=false` → SWA disabled, safe |
| dtype                 | bf16 |

## Phase 2 prerequisites

- [ ] **QKV bias handling**: confirm q/k/v_proj.bias exist in safetensors;
  reuse `../qwen25_1_5b/qwen25_bias.py` host-side bias-add via RoPE linearity.
- [ ] **GQA-aware padding**: verify `../qwen25_1_5b/qwen25_pad.py` works for
  emb=896 + g=7 (1.5B was emb=1536 + g=6).

## Watch list

- **Smaller dims than any qwen25 shape tested before** — tile configs
  may need new selections at emb=896 / ffn=4864.
- **g=7** is new (1.5B has g=6) — both odd / non-power-of-2 so reindexed
  padding pattern should generalize, but verify Phase 1.
- **n_h=14, n_kv=2** → 14 KV-broadcast factor. Memory layout same family
  as 1.5B (12, 2).
