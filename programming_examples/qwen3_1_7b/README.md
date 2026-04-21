# Qwen3-1.7B BF16 Inference on NPU2

End-to-end `Qwen/Qwen3-1.7B` inference on AMD NPU2 (AIE2P).

**Status**: Production-ready (validated 2026-04-21, independently
audited PASS-with-warnings — see [`docs/evaluation_report.md`](docs/evaluation_report.md)).

**Second validation** of the kernel-first methodology developed during
[`../qwen3_0_6b/`](../qwen3_0_6b/). Same Qwen3 architecture (Q/K Norm,
no QKV bias, GQA group=2, head_dim=128); 2.7× the parameters
(emb_dim 1024→2048, hidden_dim 3072→6144). Most code is reused unchanged
from `../llama3/`, `../_llm_shared/`, and the qwen3-0.6B scaffolding —
only shape constants change.

| Stage | Wall (warm) | Notes |
|---|---|---|
| NPU prefill @ seq_len=2048 | **2.81 s** (100 ms/layer × 28) | 3 ELFs: rms_attn_gemms (split for Q/K Norm) + flash_attn (head-first hd=128) + o_ffn |
| NPU decode | **0.149 s/token (6.7 tok/s)** | 3 fused ELFs: rms_attn_gemvs_qknorm_rope (8 launches) + o_gemv_ffn_silu (8 launches, llama3 2-K rename) + lm_head_gemv (19 partitions × 8192) |
| End-to-end (8 tokens) | ~4 s | NPU prefill + NPU LM head + 7 NPU decode tokens |

Decode is ~37% slower than the 0.6B sibling (10.5 → 6.7 tok/s) — expected
given hidden_dim doubled to 6144 (FFN dominates per-layer wall) and
emb_dim doubled to 2048 (LM head + o_proj).

## Model config

28 layers, emb_dim=2048, n_heads=16, head_dim=128, n_kv_heads=8 (GQA group=2),
hidden_dim=6144, vocab=151936, rope_θ=1e6, BF16, tied embeddings.

**Qwen3 family extensions** (vs Qwen2.5):
- NEW per-layer Q/K Norm (per-head RMSNorm BEFORE RoPE) — fused into the
  `rms_attn_gemvs_qknorm_rope` ELF, not a host op
- NO QKV bias (Qwen3 dense uses `attention_bias=False`)
- Tied embeddings, but `lm_head.weight` also stored explicitly in safetensors

**Vs qwen3-0.6B**: only shape constants change. **Crucially q_dim ==
emb_dim** for 1.7B (2048 == 2048), which means the 3-K matvec rename
collision that 0.6B needed (Q/K/V/O at K=2048, Gate/Up at K=1024,
Down at K=3072) collapses to **2 K values** for 1.7B (Q/K/V/O/Gate/Up at
K=2048, Down at K=6144). Result: llama3's existing `o_gemv_ffn_multi`
builder works directly — no qwen3-specific 3-K fork needed for 1.7B.

## Quick start

```bash
cd programming_examples/qwen3_1_7b
make compile          # one-time kernel compilation (~3 min total, cached to disk)
make run              # NPU prefill + NPU decode end-to-end
make verify N_TOKENS=8 # NPU vs CPU per-token top-1/top-5 match
make sweep            # 6/6 canonical prompt PASS check
make profile          # prefill perf measurement
```

## Documentation

| Doc | Content |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Architecture overview + 1.7B-specific config decisions |
| [TODO.md](TODO.md) | Phase status, key 1.7B-specific decisions |
| [`docs/development_progress/progress.md`](docs/development_progress/progress.md) | Phase results table + methodology validation log |
| [`docs/evaluation_report.md`](docs/evaluation_report.md) | Independent audit (re-derived correctness gates + perf) |
| [`../qwen3_0_6b/CLAUDE.md`](../qwen3_0_6b/CLAUDE.md) | First Qwen3 deployment (full Q/K Norm + 3-K rename design) |
| [`../llama3/CLAUDE.md`](../llama3/CLAUDE.md) | Canonical kernel sequence, multi-launch design |
| [`../llama32_3b/CLAUDE.md`](../llama32_3b/CLAUDE.md) | head_dim=128 + Option C FA wrapper reference |
