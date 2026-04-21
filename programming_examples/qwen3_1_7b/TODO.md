# Deployment: Qwen3-1.7B (kernel-first methodology, 2nd validation)

## Phase status
- [x] 0 Bootstrap — corr=0.99999986 vs HF, top-1 ' Paris'
- [x] 1 Per-kernel shapes — 12/12 PASS
- [x] 2 Single-block correctness — cos_real=0.9985, per-pos min > 0.98
- [x] B Step 1 fused decode #1 (rms_attn_gemvs_qknorm_rope) — cos > 0.9999
- [x] B Step 2 fused decode #2 (o_gemv_ffn_silu, llama3 builder) — cos > 0.997
- [x] 3 Full model verify (N=8) — 8/8 top-5, 7/8 exact
- [x] 3 Canonical sweep — 6/6 PASS
- [x] 4 Prefill perf — 2.81 s warm @ seq_len=2048 (100 ms/layer)
- [x] 5 Decode perf — 6.73 tok/s (148.6 ms/token)
- [x] 6 Finalize — progress.md complete; pattern reused unchanged from llama3

## Active blockers
None — deployment complete.

## Audit
Independently evaluated 2026-04-21: PASS-with-warnings. Report: docs/evaluation_report.md

## Resolved config (Qwen3-1.7B from HF)
n_layers: 28, emb_dim: 2048, n_heads: 16, n_kv_heads: 8 (GQA group=2),
head_dim: 128, hidden_dim: 6144, vocab_size: 151936, rope_θ: 1e6,
qkv_bias: False, qk_norm: True, tied embeddings: True

## Key 1.7B-specific config decisions
- LM head GEMV: 19 partitions × 8192 (vs 0.6B's 10 × 16384). At
  emb_dim=2048 the standard 16/16/8 tile config breaches L2 by 256 B;
  halving M_part to 8192 with tile_m=8 keeps both A_l2 and per-partition
  B-DMA fires within budget.
- o_gemv_ffn Down: down_tile_m=2, down_m_input=1 to fit K=6144 in L2.
- o_gemv_ffn ELF: llama3's 2-K rename builder works directly because
  q_dim==emb_dim (no need for the qwen3-0.6B 3-K fork).
