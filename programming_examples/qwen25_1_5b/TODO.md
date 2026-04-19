# Deployment: qwen25_1_5b  (Qwen/Qwen2.5-1.5B)

## Phase status
- [x] 0: Bootstrap  (2026-04-18 — top-1 " Paris", corr 0.99999992 vs HF F32)
- [x] 1: Per-kernel shapes  (2026-04-18 — classification + variant audit; 3 NEW items + 1 risk for Phase 2)
- [x] 2: Single block  (2026-04-19 — cosine 0.9988 / per-pos 0.9981 @ seq_len=2048 via GQA-reindexed padding; bias via host post-add)
- [x] 3: Full model  (2026-04-19 — decisive 3/3 top-1, competitive 3/3 top-5 overlap @ seq_len=2048)
- [x] 4: Prefill perf  (2026-04-19 — 5/5 patterns; warm 2.4 s NPU layers / 4.1 s wall via NPU FA Option C; 4.2× vs CPU-attn)
- [x] 5: Decode perf  (2026-04-19 — 216 ms/token, 4.6 tok/s, 5/6 NPU/CPU match; 5/5 patterns; new `k_split` knob in matvec.py)
- [ ] 6: Finalize

## Active blockers
(none yet)

## Phase 3 prerequisites (from Phase 2)

1. ~~**`rms_gemms_rope` ELF split for seq_len=2048**~~ — RESOLVED 2026-04-19
   via `qwen25_pad.py` GQA-aware reindexed padding. Both ELFs compile and
   produce correct output at seq_len=2048 (Phase 2 cosine = 0.9988).

2. **Inference runner must pad short prompts to seq_len ≥ 512** —
   `tile_m*herd_m=512` constraint still applies to the padded path. Document
   in inference.py and Makefile help.

3. **NPU FA at GQA group=6** still untested. Attempt Option C head-first
   wrapper at padded shapes (n_heads=16, n_kv_heads=2 → group=8 in the
   padded view; with reindex, real Q heads still group with their orig KV).
   The Option C wrapper transposes to head-first layout — phantom Q heads
   contribute zero. If it fails, follow `debug-fa-runtime-failure` recipe.

## Phase 2 prerequisites (from Phase 1) — RESOLVED IN PHASE 2

- [x] QKV bias (Qwen2 feature) — landed via `qwen25_bias.py` host post-add
      using RoPE linearity trick (no shared-code changes).
- [ ] LM Head partition for vocab=151936 — deferred to Phase 3.
- [ ] `mv_k8960.o` renamed kernel — deferred to Phase 5 (decode only).

## Original Phase 2 prerequisites text (kept for reference):

1. **QKV bias** (Qwen2 feature): extend `rms_gemms_rope_multi.py` with a
   `qkv_bias` flag that emits broadcast-add launches (bq/bk/bv 1-D over M axis)
   between each Q/K/V GEMM and the corresponding RoPE op. Keep llama3
   `qkv_bias=False` default for backward compatibility.
2. **LM Head partition for vocab=151936**: build with
   `n_partitions=10` (10 × 16384 = 163840, pad 11904). Verify
   `lm_head_multi.build_lm_head_module` and `lm_head_gemv_multi` accept 10.
3. **`mv_k8960.o` renamed kernel**: copy `_ensure_mv_k8192_o` codepath
   from `llama3_decode.py`, parameterize on K=8960.
4. **NPU FA at group=6 risk**: if Phase 2 single-block FA fails at
   `(n_heads=12, n_kv_heads=2, lq=lk=2048, dk=128)`, follow
   `debug-fa-runtime-failure` recipe.

## Resolved config (pulled from HF)
- n_layers: 28
- emb_dim (hidden_size): 1536
- n_heads: 12
- n_kv_heads: 2  (GQA group_size = 6)
- head_dim: 128
- hidden_dim (intermediate_size): 8960
- vocab_size: 151936
- rope_theta: 1,000,000
- dtype: BF16
- tie_word_embeddings: True
- **qkv_bias: True**  ← NEW architectural feature vs prior deployments

## Anticipated tier
**Tier-C** (one new feature: QKV bias). Other features already covered:
- head_dim=128 → reuse Option C head-first FA wrapper from `llama32_3b/`
- tied embeddings → reuse `smollm2_1_7b/` weight loader pattern
- GQA → reuse `llama3/` multi-launch builder

Estimated effort: 1-2 days end-to-end if QKV-bias kernel addition is small.
