# Qwen2.5-1.5B BF16 Inference on NPU2 (MLIR-AIR)

End-to-end `Qwen/Qwen2.5-1.5B` inference on AMD NPU2 (AIE2P).

## Status

**Deployed and operational** (2026-04-19). All 7 phases of `deploy-new-llm`
PASSED.

## Performance (validated)

| Phase | Metric |
|---|---|
| Prefill (warm, NPU FA Option C) | **2.4 s NPU layers** (85 ms/layer, seq_len=2048) |
| Decode steady-state | **216 ms/token (4.6 tok/s)** |
| Per-layer rate | **7.7 ms/layer** — matches llama32_3b at same depth + head_dim |
| End-to-end (`make run`, 14 tokens) | **6.1 s wall** |
| Top-1 NPU/CPU match (decode) | 5/6 (>80% gate) |

Generated text from `make run`:

```
"The capital of France is Paris, the capital of the United Kingdom is
 London, and the capital of"
```

## Quick start

```bash
cd programming_examples/qwen25_1_5b
make compile          # One-time kernel compilation (~2-3 min, cached)
make run              # End-to-end demo: NPU prefill + NPU decode
make run N_TOKENS=20  # Generate 20 tokens
make profile          # Prefill perf measurement (cold + warm)
make verify           # Decode with NPU/CPU top-1 verification
```

## Model config

28 layers, emb_dim=1536, n_heads=12, **head_dim=128**, n_kv_heads=2 (GQA
group=6), hidden_dim=8960, vocab=151936, BF16, **rope_θ=1,000,000**,
tied embeddings, **QKV bias = True**.

## Two architectural firsts handled by this deployment

1. **QKV bias** (Qwen2 family) — added on the HOST after the bias-free
   `rms_gemms_rope` ELF returns, exploiting RoPE's linearity:
   `RoPE(q + bq) = RoPE(q) + RoPE(bq)`. Implementation:
   [`qwen25_bias.py`](qwen25_bias.py).

2. **emb_dim=1536 + hidden_dim=8960 not BD-friendly** — neither is a
   clean multiple of 1024, so the AIE shim DMA can't pack them as
   single-dim BDs. Two consequences:
   - Prefill at seq_len=2048 hit BD pool exhaustion → fixed by
     **GQA-aware reindexed padding** (emb_dim 1536→2048, hidden_dim
     8960→9216, Q heads inserted INSIDE each KV group to preserve GQA
     group_size). [`qwen25_pad.py`](qwen25_pad.py).
   - Decode Down GEMV at K=8960 hit two `repeat_count > 255` walls →
     fixed by `tile_m=16, m_input=16` for M=8960-class GEMVs and the
     new **`k_split=70`** matvec parameter (additive, back-compat).
     [`qwen25_decode_setup.py`](qwen25_decode_setup.py).

## Pipeline

```
embed (CPU)
--- prefill (per prompt) — PADDED shapes (emb=2048, hidden=9216) ---
-> 28× [rms_gemms_rope (NPU 6-launch ELF, host bias post-add) ->
        NPU FA (Option C head-first wrapper at head_dim=128) ->
        o_ffn (NPU 8-launch ELF)]
   extracting per-layer K (post-RoPE) and V into KV cache
-> slice last hidden state from padded emb_dim back to orig (1536)
-> CPU final RMSNorm at last prompt position
-> NPU LM Head GEMV (10×16384 partition, vocab=151936)
-> argmax -> first generated token
--- decode (per token) — ORIG shapes (emb=1536, hidden=8960) ---
-> 28× [rms_gemv_rope (NPU 6-launch ELF, host bias via decode-pos) ->
        CPU attention (KV cache) ->
        o_gemv_ffn (NPU 8-launch ELF, mv_k8960.o + k_split=70)]
-> CPU final RMSNorm + NPU LM Head GEMV -> argmax
```

## File structure

| File | Purpose |
|---|---|
| `qwen25_inference.py` | End-to-end NPU runner — entry point for `make run` |
| `qwen25_weights.py` | HF safetensors loader (incl. QKV bias) + RoPE LUT |
| `qwen25_reference.py` | CPU F32 reference forward pass (bias + Qwen2 eps) |
| `qwen25_pad.py` | GQA-aware reindexed padding (prefill) |
| `qwen25_bias.py` | RoPE-linearity host bias add (cache-level monkey-patch) |
| `qwen25_decode_setup.py` | mv_k8960.o + 10-partition LM head + decode helpers |
| `qwen25_phase{2..5}_test.py` | Per-phase validation drivers |
| `Makefile` | Build/run/profile/verify targets |

## Documentation

| Doc | Content |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Model-specific guide for Claude Code |
| [TODO.md](TODO.md) | Phase status (all PASSED) |
| [docs/development_progress/](docs/development_progress/) | Per-phase results, lessons, debug log |
| [docs/development_progress/phase6_finalize.md](docs/development_progress/phase6_finalize.md) | End-to-end perf summary + reusable-pattern audit |
| [docs/development_progress/LESSONS.md](docs/development_progress/LESSONS.md) | 5 lessons from this deployment |
| [`../llama3/CLAUDE.md`](../llama3/CLAUDE.md) | Canonical kernel sequence, multi-launch design |
| [`../llama32_3b/CLAUDE.md`](../llama32_3b/CLAUDE.md) | head_dim=128 + Option C FA wrapper reference |

## Comparison vs prior deployments (warm steady-state)

| Model | n_layers | head_dim | Prefill ms/layer | Decode tok/s | Decode ms/layer |
|---|---|---|---|---|---|
| llama3 (1B) | 16 | 64 | 81 | 10.8 | 5.75 |
| smollm2 (1.7B) | 24 | 64 | 79 | 7.3 | 5.70 |
| llama32_3b (3B) | 28 | 128 | 115 | 4.7 | 7.7 |
| **qwen25_1_5b** | **28** | **128** | **85** | **4.6** | **7.7** |

Per-layer rates match the family despite the architectural firsts (QKV
bias + non-1024-aligned dims) — confirms the padding/reindex/bias paths
add **zero runtime overhead**.
