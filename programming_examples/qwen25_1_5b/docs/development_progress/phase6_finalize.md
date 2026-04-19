# Phase 6 — Finalize: Qwen2.5-1.5B end-to-end NPU2 deployment

**Date**: 2026-04-19
**Status**: PASS — end-to-end NPU runner wired (`qwen25_inference.py` + `make run`).

## End-to-end demo (`make run` / `python3 qwen25_inference.py`)

Prompt: `'The capital of France is'`
Tokens generated: 14 (after the seeded first token)

```
"The capital of France is Paris, the capital of the United Kingdom is
 London, and the capital of"
```

Coherent, factually correct, multi-fact continuation.

| Metric | Value |
|---|---|
| Compile / cache load (one-time) | 3.0 s |
| Weight load + pad-helper (one-time) | 3.6 s |
| BO preload + bias registry (one-time) | 6.0 s |
| **NPU prefill** (28 layers, padded shapes, NPU FA Option C) | **2.59 s** (92 ms/layer) |
| First LM Head GEMV | 205 ms |
| **Decode** (14 tokens) | avg **237 ms/token (4.2 tok/s)**, median 203 ms/token |
| **Total inference wall** (prefill + 14 decode) | **6.11 s** |

## Per-phase summary (Phases 0–6)

| Phase | Outcome | Key metric |
|---|---|---|
| 0 — Bootstrap | PASS | CPU reference cosine **0.99999992** vs HF F32; top-1 ` Paris` |
| 1 — Per-kernel shapes | PASS | classification + variant audit; 3 NEW work items + 1 risk surfaced |
| 2 — Single block | PASS | whole-tensor cosine **0.9988**, per-pos min **0.9981** @ seq_len=2048 |
| 3 — Full model | PASS | decisive 3/3 top-1; competitive 3/3 top-5 overlap; no NaN |
| 4 — Prefill perf | PASS | 5/5 patterns; **2.4 s warm** (85 ms/layer), 4.2× vs CPU-attn |
| 5 — Decode perf | PASS | 5/5 patterns; **216 ms/token (4.6 tok/s)**, 7.7 ms/layer |
| 6 — Finalize | PASS | end-to-end runner wired; coherent multi-fact text generation |

## Comparison vs prior deployments (ALL deployed via the same skill chain)

| Model | n_layers | head_dim | Warm prefill (NPU layers) | ms/layer (prefill) | Decode | tok/s | ms/layer (decode) |
|---|---|---|---|---|---|---|---|
| llama3 (1B) | 16 | 64 | 1.30 s | 81 | 92 ms/tok | 10.8 | 5.75 |
| smollm2 (1.7B) | 24 | 64 | 1.88 s | 79 | 137 ms/tok | 7.3 | 5.70 |
| llama32_3b (3B) | 28 | 128 | 3.2 s | 115 | 215 ms/tok | 4.7 | 7.7 |
| **qwen25_1_5b** | **28** | **128** | **2.4 s** | **85** | **216 ms/tok** | **4.6** | **7.7** |

Qwen2.5-1.5B's per-layer rates **match the family**:
- Prefill ms/layer: **85** — between smollm2's 79 (head_dim=64) and llama32_3b's 115 (head_dim=128, emb_dim=3072). Smaller emb_dim=1536 keeps it lean.
- Decode ms/layer: **7.7** — exact match to llama32_3b (same n_layers + head_dim).

This confirms our padding/reindex/bias overhead is **negligible** at runtime — the kernels run as if natively shaped.

## Reusable infra promoted to shared this deployment

Three additive, back-compat-default helpers landed during qwen25_1_5b
(verified to not affect llama3/smollm2/llama32_3b):

1. **`matvec.build_module(k_split=...)`**
   (`programming_examples/matrix_vector_multiplication/bf16/matvec.py`)
   Pre-splits the K-DMA dim so the lowering doesn't auto-split into a
   `(K/32, 32)` BD chain that exceeds the AIE2P shim's 255-firing limit.
   Default `None` preserves byte-identical IR. Reusable for ANY future
   model with `hidden_dim > 8160` (Llama-3-8B's 14336 will need this too).

2. **`build_o_gemv_ffn_module(down_k_split=...)`**
   (`programming_examples/llama3/multi_launch_builder/o_gemv_ffn_multi.py`)
   Forwards `k_split` to the Down GEMV's matvec call. Default `None`.

3. **`KernelCache.load_and_run` cache-level monkey-patch pattern**
   (`programming_examples/qwen25_1_5b/qwen25_bias.py`)
   Catches BOTH prefill (`rms_gemms_rope`) and decode (`rms_gemv_rope`)
   for layer-aware host-side post-add. Works for any "linear-with-RoPE"
   transformation that an existing bias-free ELF can absorb on the host.

## Qwen2-family-specific insights (NOT yet promoted, but ready)

These two helpers are Qwen2-family-specific and live in `qwen25_1_5b/`.
If we deploy Qwen2.5-3B / 7B / Qwen3-* next, they should be promoted
to `_llm_shared/`:

- **`qwen25_pad.py`**: GQA-aware reindexed padding. Pads emb_dim and
  hidden_dim to BD-friendly multiples of 1024, INSERTING phantom Q
  heads INSIDE each KV group to preserve GQA mapping. Reusable for any
  model whose dims aren't BD-friendly.
- **`qwen25_bias.py`**: RoPE-linearity QKV bias on host. Reusable for
  any Qwen2-family model (or any model that adds a 1-D bias before
  RoPE).

## Lessons captured in this deployment (LESSONS.md)

| # | Lesson | Reusable for |
|---|---|---|
| 1 | RoPE-linearity host bias add | any Qwen2-family model |
| 2 | GEMM tile config silently corrupts at small N or M | any new shape combo |
| 3 | BD pool exhaustion at non-1024-aligned emb_dim + multi-launch ELF | any model with emb_dim ∉ {1024k} |
| 4 | Padding emb_dim breaks GQA without head-axis reindex | any GQA model needing dim padding |
| 5 | Two distinct repeat_count > 255 walls at hidden_dim ≥ ~8200 | any model with hidden_dim > 8160 |

## Open follow-ups (not blocking)

1. Promote `qwen25_pad.py` and `qwen25_bias.py` to `_llm_shared/` if a
   third Qwen2-family or padding-needing deployment lands.
2. Move attention to NPU during decode (Pattern 5 currently `[PARTIAL]`,
   matches llama3 design — would unlock additional ~20-40 ms/token if
   wired).
3. Consider exposing `tile_k_l1` in matvec.py for finer-grained control
   over K-loop count (current K_max via auto-split is 32×255=8160; with
   `tile_k_l1=64` would extend to 64×255=16320).

## Memory footprint

Approximate peak runtime working set on NPU2's 16 GB DRAM:
- ~3.0 GB CPU-side BF16 weights (orig)
- ~3.5 GB padded prefill BOs (28 layers × per-layer weight BOs)
- ~3.5 GB decode BOs (28 layers + 10 LM head partitions)
- ~few hundred MB precomputed bias tensors

Total ~10 GB working set — comfortably within the 16 GB budget.

**Deployment complete.**
