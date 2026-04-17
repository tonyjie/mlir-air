# Deployment: llama32_1b_instruct

Smoke test of the `deploy-new-llm` skill chain on `meta-llama/Llama-3.2-1B-Instruct`.
Architecture is identical to `Llama-3.2-1B` (already deployed at `programming_examples/llama3/`),
differing only in fine-tuned weights — so most phase gates are expected to pass trivially.

## Phase status
- [x] 0: Bootstrap (PASSED 2026-04-17 — reference top-1 " Paris", prob=0.699)
- [x] 1: Per-kernel shapes (PASSED by reference — same shapes as llama3 base)
- [x] 2: Single block (PASSED by reference — same kernels as llama3 base)
- [x] 3: Full model (PASSED 2026-04-17 — measured: "A: The capital of France is Paris.")
- [x] 4: Prefill perf (PASSED — 1.53s, same as llama3 base via symlinked cache)
- [x] 5: Decode perf (PASSED — 92 ms/token, same as llama3 base)
- [x] 6: Finalize (PASSED — see docs/development_progress/progress.md)

## Active blockers
(none — deployment complete)

## Resolved config (identical to Llama-3.2-1B base)
n_layers: 16, emb_dim: 2048, n_heads: 32, n_kv_heads: 8,
head_dim: 64, hidden_dim: 8192, vocab_size: 128256, rope_theta: 500000.0
