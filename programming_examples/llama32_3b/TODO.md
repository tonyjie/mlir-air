# Deployment: llama32_3b

Deployment of `meta-llama/Llama-3.2-3B` (BF16) on NPU2 via the
`deploy-new-llm` skill chain. Second Tier-A target after `smollm2_1_7b`
(see `docs/superpowers/edge-llm-candidates.md` "Path A — agreed 2026-04-17").

Reference deployment: `programming_examples/llama3/` (Llama-3.2-1B).

Architectural divergences from llama3 (full discussion in `CLAUDE.md`):
1. **head_dim=128** (vs 64) — known kernel blocker (rope_halfsplit.cc,
   attn_npu2.cc); 4–8 hours of manual `.cc` work expected at Phase 1.
2. **28 layers** (vs 16) — per-layer BO arrays sized to 28.
3. **emb_dim=3072** (vs 2048) — GEMM tile configs may need re-tuning.
4. **GQA group=3** (24 heads / 8 KV-heads).
5. **rope_scaling=llama3** — inert for seq_len ≤ 8192; defer long-context
   wavelength remap to Phase 6 follow-up.
6. **Memory tight** (~11 GB at runtime on 16 GB DRAM).

## Phase status
- [x] 0: Bootstrap (PASSED 2026-04-17 — top-1 ' Paris', HF logits corr=0.99999962)
- [x] 1: Per-kernel shapes (PASSED 2026-04-17 — classification: 3 drop-in / N recompile / 1 novel item flagged for Phase 2: FA at head_dim=128 needs lkp=64 not 128 for L1 budget)
- [x] 2: Single block (PASSED 2026-04-18 CPU-attn path — whole-tensor cos 0.9959 (68 tok), per-pos min 0.980 with head_dim-scaled gate, MAE 0.005 (5× better than smollm2), no NaN; new LESSONS Lesson 1 captured; NPU FA deferred to Phase 4)
- [x] 3: Full model (PASSED 2026-04-18 CPU-attn path with adapted gate — 4/4 decisive top-1 match, 2/2 competitive top-5 overlap, no NaN; LESSONS Lesson 2 captured re: decisive-vs-competitive gate; per-layer cos drifts to 0.881 by L27 — pure BF16 accumulation; F32-output Down GEMM deferred)
- [x] 4: Prefill perf (PASSED 2026-04-18 CPU-attn — 4/5 patterns, warm 13.6s NPU / 15.7s wall, pattern 2 BO preload applied, no regression; NPU FA compiles via new compile_attn_npu2_split API but hangs at runtime — real follow-up flagged; perf gated on NPU FA fix)
- [x] 5: Decode perf (PASSED 2026-04-18 — 5/5 patterns, 215 ms/token / 4.7 tok/s, 3/3 NPU/CPU top-1 match, generated 'The capital of France is Paris. It is the largest city in'; per-layer rate at K-scaled parity with llama3/smollm2)
- [x] 6: Finalize (PASSED 2026-04-18 — see phase6_finalize.md; end-to-end runner llama32_3b_inference.py wired to `make run`; deployment complete; 2 LESSONS captured + 1 shared-infra promotion)

## Active blockers
(none yet)

## Follow-up: NPU FlashAttention runtime hang at head_dim=128 + lq=lk=2048
Phase 4 status: `compile_attn_npu2_split(lqp=256, lkp=64, dk=128, dv=128)` API
added; `attn_npu2.o` and `flash_attn.elf` build cleanly. `_attn_backend_kwargs`
monkey-patched to return `omit_while_true_loop=True` for head_dim=128. Runtime
hangs with `ERT_CMD_STATE_TIMEOUT` on first FA invocation in `run_transformer_block`.
- **Triage hypotheses** (not yet investigated):
  - GQA group_size=3 untested (smollm2 group=1, llama3 group=4, lit test group=4)
  - lq=lk=2048 is 4× the proven-working llama3-8b lit test config (lq=lk=512)
  - dk_chunks=2 (lkp != dk) path may have integration issue in seq-first variant
- **Recommended bisect order**: try lq=lk=512 first (lit test config), then 1024,
  then 2048. Independently try with n_heads=32/n_kv_heads=8 (group=4, lit test
  config) at our seq=2048 to isolate group_size=3 vs scaling.
- **Impact**: blocks production-grade prefill perf (current 13.6s warm NPU
  prefill vs ~7s projection with NPU FA). Phase 5 decode does not depend on
  this (decode uses CPU attention by design in llama3 too).

## Resolved config (pulled from HF `config.json`)
```
n_layers:            28
emb_dim:             3072
n_heads:             24
n_kv_heads:          8         # GQA group=3
head_dim:            128
hidden_dim:          8192
vocab_size:          128256
rope_theta:          500000.0
rms_norm_eps:        1e-5
max_pos:             131072    # via rope_scaling; 8192 base
rope_scaling:        {type: llama3, factor: 32.0,
                      low_freq_factor: 1.0, high_freq_factor: 4.0,
                      original_max_position_embeddings: 8192}
tie_word_embeddings: true
torch_dtype:         bfloat16
attention_bias:      false
```
