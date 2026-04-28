# Deployment: qwen25_3b  (Qwen/Qwen2.5-3B)

## Phase status
- [x] 0: Build CPU Oracle  (PASS 2026-04-27, 3.6 min wall — per-layer cos min 0.999999 all 36L, final cos 1.00, top-1 ' Paris' match HF)
- [x] 1: Kernel Validation (PASS 2026-04-27, 6.1 min wall — 14/15 PASS standalone + Down GEMV K=11008 deferred to Phase 5; min cos 0.994 FA hd=128, max 0.999996)
- [x] 2: Single-Block Validation (PASS 2026-04-27, 31.2 min wall — CPU FA cos 0.998 / NPU FA Option C cos 0.997 + per-pos 0.995. **W1 NOT reproduced** at hd=128 — finding refines W1 as seq-first-FA-specific. Padded hidden=12288 (BD-friendly).)
- [x] 3: Full-Model Validation (PASS 2026-04-27, 5.9 min wall — 6/6 prompts top-1 match: 4/4 decisive + 2/2 competitive. ' Paris' / ' Pacific' / ' divided' all correct. ~102 ms/layer prefill across 36L)
- [x] 4: Prefill Optimization (PASS 2026-04-27, 2.2 min wall — 5/5 patterns. Warm 103 ms/layer ± 122 ms; cold→warm B1 gain 49.7%. Wall 5.44 s. Top-1 ' Paris' preserved across 5 runs)
- [x] 5: Decode Optimization (PASS 2026-04-27, 13.7 min wall — 5/5 patterns. Steady 344 ms/token = 2.9 tok/s = 9.6 ms/layer; **NEW partition scheme 11×13824** vs 1.5B's 10×16384 due to Rule D L2 cap at K=2048; mv_k11008.o + down_k_split=86; 4/4 CPU verify match)
- [⚠] 6: Finalize & Learn  (PASS-with-warnings 2026-04-27, 31.9 min wall — inference.py + Makefile targets work. **W2 workaround (CPU LM head) landed**: 5/5 NPU/CPU greedy match in `make verify`.)
- [⚠] 7: Independent Evaluation  (PASS-with-warnings 2026-04-27, 15.3 min wall — evaluator confirms make verify 5/5 PASS + 2/2 reproducibility + 3/3 adversarial. **REFINED W2**: Phase 5 NPU LM head is FLAKY (3 runs: 0/4→3/3→0/4), not always-works. Paper should cite inference.py CPU-LM-head decode perf (240 ms/tok median) not Phase 5 NPU LM head 344 ms/tok.)

## Active blockers / warnings

### W2 (2026-04-27, Phase 6) — NPU LM Head broken via inference.py path

**Symptom**: `make run` and `make verify` produce garbage output (NPU LM
Head returns argmax=0 or junk tokens). Phase 5 phase5_test with CPU prefill
seed produces correct output (4/4 NPU/CPU match) using THE SAME code paths.

**Differential**: Only difference between Phase 5 (works) and Phase 6
(broken) is whether NPU prefill ran before LM Head usage. Suspected: NPU
prefill leaves shared state (BO, channel, or k_cache layout) that
contaminates subsequent NPU LM head GEMV calls.

**Workaround tried**: CPU LM head for first_token (fixed first_token but
decode loop NPU LM head still broken).

**Hypothesis tested + failed**: Re-call `preload_qwen25_3b_lm_head` before
LM head usage. Did NOT fix. Tried explicit cache._loaded reset too — also
failed.

**Workaround in place**: CPU LM head for ALL inference.py calls (`make verify`
5/5 PASS via this).

**Paper-relevant data (REVISED after Phase 7 audit)**:
- Correctness: Phase 3 top-1 6/6 (NPU prefill correct), Phase 6 make verify
  5/5 with CPU LM head (NPU prefill + NPU decode correct, only LM head needs
  CPU). NPU LM head 11×13824 ELF is **flaky / not citable**.
- Performance: Phase 4 NPU prefill 103 ms/layer warm (citable), Phase 6
  decode median 240 ms/token via CPU LM head (citable, reproducible).
  Phase 5 NPU-LM-head 344 ms/tok number was from a single lucky run; do
  NOT cite as steady-state perf.

## Resolved config (from HF — `Qwen/Qwen2.5-3B`)

| Field | Value |
|---|---|
| n_layers              | **36** (deepest deployment in catalog) |
| emb_dim (hidden_size) | 2048 (1024-aligned ✓ — no emb padding) |
| n_heads               | 16 (even ✓ — FA OK) |
| n_kv_heads            | 2 → GQA group g=**8** (same as qwen25_0_5b — W1 risk) |
| head_dim              | **128** (Option C head-first FA wrapper required) |
| ffn_hidden            | **11008** (NOT 1024-aligned; pad to 11264) |
| vocab                 | 151936 (same as qwen25_1_5b → 10×16384 LM head partition) |
| rope_theta            | 1000000 (same) |
| tie_word_embeddings   | true |
| attention_bias        | (Qwen2 default → True) |
| sliding_window        | 32768 set, `use_sliding_window=false` → SWA disabled |
| dtype                 | bf16 |

## Phase 2 prerequisites

- [ ] **QKV bias**: reuse `../qwen25_1_5b/qwen25_bias.py` (model-agnostic)
- [ ] **GQA-aware padding** for hidden 11008→11264: reuse
  `../qwen25_1_5b/qwen25_pad.py` with `padded_emb_dim=2048`,
  `padded_hidden_dim=11264`, no n_h pad needed
- [ ] **Option C head-first FA**: reuse `_llm_shared/phase_helpers/headfirst_fa.py`
- [ ] **Down K-split**: write `qwen25_3b_decode_setup.py` with
  `mv_k11264.o` + `down_k_split=88` (88×128=11264)

## Watch list

- **W1**: NPU FA precision drop at GQA g=8 + n_kv=2 — qwen25_0_5b hit
  cos 0.978/per-pos 0.94 in single-block test. Same GQA shape here →
  expect same warning. Top-1 should still match (qwen25_0_5b did 6/6
  + 3/3 adversarial).
- **36 layers compile time** (deepest deployment) — may need patience.
- **Down K=11264**: new K-split factor = 88 (split into 88 outer ×
  128 inner; satisfies Rule B repeat ≤ 255 and Rule D L2 cap).
