# Qwen3-0.6B BF16 Inference on NPU2

End-to-end `Qwen/Qwen3-0.6B` inference on AMD NPU2 (AIE2P).

**Status**: Production-ready (validated 2026-04-20).

| Stage | Wall (warm) | Notes |
|---|---|---|
| NPU prefill @ seq_len=2048 | **2.09 s** (74.8 ms/layer × 28) | 3 fused ELFs: rms_gemms (split for Q/K Norm) + flash_attn (head-first hd=128) + o_ffn |
| NPU decode | **0.09 s/token (10.7 tok/s)** | 3 fused ELFs: rms_attn_gemvs_qknorm_rope (8 launches) + o_gemv_ffn_silu (8 launches, 3-K matvec rename) + lm_head_gemv (10 partitions) |
| End-to-end (8 tokens) | ~3 s | NPU prefill + NPU LM head + 7 NPU decode tokens |

Per-layer parity with llama3-1B (10.8 tok/s) despite 28 layers vs 16,
head_dim=128 vs 64, and the Qwen3-specific Q/K Norm + 3-K matvec extern.

## Model config

28 layers, emb_dim=1024, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
hidden_dim=3072, vocab=151936, rope_θ=1e6.

**Qwen3 family extensions** (vs Qwen2.5):
- NEW per-layer Q/K Norm (per-head RMSNorm BEFORE RoPE) — fused into the
  `rms_attn_gemvs_qknorm_rope` ELF, not a host op
- NO QKV bias (Qwen3 dense uses `attention_bias=False`)
- Tied embeddings (but `lm_head.weight` also stored explicitly in safetensors)

## Quick start

```bash
cd programming_examples/qwen3_0_6b
make compile          # one-time kernel compilation (~3 min total)
make run              # NPU prefill + NPU decode end-to-end
make verify           # NPU vs CPU decode per-token top-1 match (qwen3_verify_decode.py)
make profile          # prefill perf measurement
```

## Documentation

| Doc | Content |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Architecture overview + divergences from prior deployments |
| [TODO.md](TODO.md) | Phase status, resolved blockers, follow-ups |
| [`docs/development_progress/phase_b_fusion.md`](docs/development_progress/phase_b_fusion.md) | Phase B fusion design + 10× host-side speedup |
| [`docs/development_progress/phase6_finalize.md`](docs/development_progress/phase6_finalize.md) | Phase 6 deployment summary (CPU-decode interim) |
| [`docs/development_progress/LESSONS.md`](docs/development_progress/LESSONS.md) | Novel failure modes encountered |
