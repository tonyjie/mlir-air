# Deployment: Qwen3-4B

## Phase status
- [x] 0: Build CPU Oracle  (PASS 2026-04-27, 36/36 layers cos 1.000000, final logits cos 0.99999983, top-1 ' Paris')
- [x] 1: Kernel Validation  (PASS 2026-04-27, 6/13 cold + 7/13 carry-over + 9 GEMV/LM-head deferred to Phase 5; min cos 0.999745, max 0.999984)
- [x] 2: Single-Block Validation  (PASS 2026-04-27, padded emb=3072 hidden=10240, seq=2048 production: cos 0.998753 / per_pos_min 0.998232; same numbers seq=512)
- [x] 3: Full-Model Validation  (PASS 2026-04-27, 6/6 canonical prompts NPU top-1 == CPU top-1; NPU 10.5s vs CPU 51s per prompt at seq=2048)
- [x] 4: Prefill Optimization  (PASS 2026-04-27, P2/P3 applied + P4 already + P1/P5 N/A; cold 13.5s → warm 8.05s = 1.68×; 224 ms/layer at 36L padded split-ELF)
- [x] 5: Decode Optimization  (PASS 2026-04-27, 3 decode ELFs compiled with per-launch tile_m fix; NPU decode 387 ms/token ≈ 2.6 tok/s; correct tokens ' Paris', '.', ' The', ' capital', ' of')
- [x] 6: Finalize & Learn  (PASS 2026-04-27, make verify ' Paris' match + make run N_TOKENS=10 generates 'The capital of France is Paris. The capital of Paris is...? The'; NPU prefill 8.00s + decode 387 ms/tok)
- [x] 7: Independent Evaluation  (PASS-with-warnings 2026-04-27 by independent-evaluator subagent: verify gate honest, 3/3 adversarial prompts top-1 match, byte-identical reproducibility; V-cache drift across 36L is informational; see docs/evaluation_report.md)

## Active blockers
(none yet)

## Resolved config (pulled from HF Qwen/Qwen3-4B)
n_layers: 36, emb_dim: 2560, n_heads: 32, n_kv_heads: 8 (GQA g=4),
head_dim: 128, hidden_dim: 9728, vocab_size: 151936,
rope_theta: 1000000, tied embeddings, NO QKV bias, Q/K Norm.

q_dim = n_heads * head_dim = 4096 (≠ emb_dim 2560 → 3-K matvec rename).

## Phase 2 prerequisites (surface from Step 2 architecture check)
- emb_dim=2560 + hidden=9728 are NOT 1024-aligned → padding required
  (apply qwen25_pad-style or kernel-first split-ELF; decide in Phase 2)
- Q/K Norm BEFORE RoPE (mirror qwen3_0_6b's host wrapper or on-tile)
- Option C head-first FA (head_dim=128)
