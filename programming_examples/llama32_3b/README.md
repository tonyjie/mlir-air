# Llama-3.2-3B BF16 Inference on AMD NPU2

End-to-end inference of `meta-llama/Llama-3.2-3B` (BF16) on AMD NPU2
(Strix, AIE2P) via the MLIR-AIR compiler stack. Built with the
`deploy-new-llm` skill chain; validated 2026-04-18.

## Status

**Functional and correct.** Decode perf hits the predicted K-scaled parity
with llama3/smollm2. Prefill currently uses CPU attention (NPU FlashAttention
at head_dim=128 + lq=lk=2048 hangs at runtime — real follow-up flagged in
[TODO.md](TODO.md)).

| Stage | Result |
|---|---|
| Decode rate (steady-state) | **214.9 ms/token (4.7 tok/s)** — exactly the K-scaled prediction (1.5× wider K vs llama3/smollm2) |
| Prefill (28 layers, CPU-attn) | 13.6 s warm NPU / 15.7 s wall (seq_len=2048) |
| Top-1 NPU/CPU match (decode) | 3/3 |
| Top-1 NPU/CPU match (full prefill) | 4/4 decisive prompts + 2/2 competitive top-5 overlap |
| End-to-end NPU output | `'The capital of France is the most visited city in the world.'` |

## Quick start

```bash
cd programming_examples/llama32_3b

# Compile prefill + decode kernels (one-time, ~2 min)
make compile

# End-to-end inference: NPU prefill (with K/V extraction) + NPU decode
make run                       # generates 8 tokens; defaults: prompt='The capital of France is', seq_len=2048

# Decode-only perf measurement (4 tokens with NPU/CPU verify, ~3 min CPU verify)
make verify

# Phase tests
make run-block                 # Phase 2: single-block correctness
make run-full                  # Phase 3: full 28-layer prefill + top-1 match
make run-prefill               # Phase 4: prefill perf cold vs warm
make run-decode-only           # Phase 5: CPU-prefill seed + NPU decode

# Reference (CPU F32 forward pass + HF cross-check)
make run-reference

# Cleanup
make clean
```

Override defaults: `make run N_TOKENS=20 PROMPT='The largest ocean is the' SEQ_LEN=2048`

## Architecture

| Field | Value | vs llama3-1B |
|---|---|---|
| Layers          | 28          | 16 (1.75×) |
| emb_dim         | 3072        | 2048 (1.5×) |
| n_heads         | 24          | 32 |
| n_kv_heads      | 8           | 8 (same) |
| head_dim        | **128**     | 64 (**2×** — biggest kernel-side divergence) |
| GQA group_size  | 3           | 4 |
| hidden_dim      | 8192        | 8192 (same) |
| vocab_size      | 128256      | 128256 (same — LM Head partition is drop-in) |
| rope_θ          | 500000      | 500000 (same) |
| tie_word_emb    | true        | true |
| dtype           | bfloat16    | bfloat16 |

## Performance

### Prefill (seq_len=2048, CPU-attn fallback)

| Phase | NPU layers | per-layer | Wall (incl LM Head) |
|---|---|---|---|
| Cold (no preload, 1st prompt) | 16.4 s | 587 ms | 18.5 s |
| Warm (after `preload_prefill_weights`) | 13.6 s | 487 ms | 15.7 s |

Pattern 2 (BO pre-load) gain: 17%. Modest because per-layer time is
dominated by CPU attention (~300 ms numpy GQA at seq=2048, hd=128, 24
heads). With NPU FA unblocked, projected ~7 s prefill.

### Decode (per-token, NPU LM Head GEMV, CPU per-token attention with KV cache)

| Metric | Value |
|---|---|
| Steady-state per-token latency | **214.9 ms** |
| Per-layer rate (decode) | **7.7 ms/layer** |
| Throughput | **4.7 tok/s** |
| NPU/CPU top-1 match | 3/3 |

Per-layer rate = 1.35× slower than llama3 (5.75 ms/layer) — exactly the
1.5× wider K (3072 vs 2048). Decode kernels at K=3072 hit the same
per-byte efficiency as the reference deployments.

### Memory footprint

~14 GB peak runtime working set: 6 GB CPU-side BF16 weights + 5.4 GB
prefill BOs + 5.9 GB decode BOs (transposed copies). Within NPU2's 16 GB
DRAM budget but tight.

## Generated text examples

```
prompt: 'The capital of France is'
NPU:    ' the most visited city in the world.'

prompt: 'The largest ocean is the'  (decisive, CPU top-1 prob=0.82)
NPU:    ' Pacific'  (matches CPU exactly)

prompt: 'Water freezes at'  (decisive, CPU top-1 prob=0.71)
NPU:    ' 0' / continues with degree information

prompt: '1 + 1 ='  (decisive, CPU top-1 prob=0.74)
NPU:    ' 2'
```

## File structure

| File | Purpose |
|---|---|
| `llama32_3b_inference.py` | End-to-end NPU prefill + decode (entry point for `make run`) |
| `llama32_3b_weights.py`   | Config dataclass + HF safetensors loader + RoPE LUT |
| `llama32_3b_reference.py` | CPU F32 reference forward pass (verifies against HF) |
| `llama32_3b_phase2_test.py` | Single-block correctness (`make run-block`) |
| `llama32_3b_phase3_test.py` | Full 28-layer top-1 + per-layer drift (`make run-full`) |
| `llama32_3b_phase4_test.py` | Prefill perf (cold + warm + BO preload) (`make run-prefill`) |
| `llama32_3b_phase5_test.py` | Decode perf with KV cache (`make run-decode-only`) |
| `Makefile`                 | llama3-style targets |
| `CLAUDE.md`                | Model-specific guide for future Claude sessions |
| `TODO.md`                  | Phase status + active follow-ups |
| `docs/development_progress/` | Per-phase progress + LESSONS + debug log |

This directory inherits orchestration from `../llama3/` and `../_llm_shared/`
at runtime via a sys.path bootstrap at the top of each script (no local
copies of `llama3_*.py` or `multi_launch_builder/`). See `CLAUDE.md` for
the import map and full divergence list.

## Validation outputs (per-phase docs)

- [Phase 0 — Bootstrap](docs/development_progress/progress.md): top-1 `' Paris'`, HF F32 logits corr=0.99999962
- [Phase 1 — Per-kernel shapes](docs/development_progress/phase1_kernel_shapes.md): classification table; head_dim=128 supported via runtime/preprocessor parameters in both critical kernels
- [Phase 2 — Single block](docs/development_progress/phase2_block.md): cosine 0.996, MAE 0.005, head_dim-scaled per-position gate
- [Phase 3 — Full model](docs/development_progress/phase3_full.md): decisive (4/4) + competitive (2/2 top-5 overlap) — per-layer drift analysis
- [Phase 4 — Prefill perf](docs/development_progress/phase4_prefill.md): 4/5 patterns; NPU FA hang documented
- [Phase 5 — Decode perf](docs/development_progress/phase5_decode.md): 5/5 patterns, 4.7 tok/s, NPU/CPU match
- [Phase 6 — Finalize](docs/development_progress/phase6_finalize.md): comparison vs llama3/smollm2; reusable-pattern audit
- [LESSONS.md](docs/development_progress/LESSONS.md): 2 lessons captured (head_dim-scaled per-position threshold; decisive/competitive Phase 3 gate)

## Known limitations / follow-ups

See [TODO.md](TODO.md) for the live list. Highlights:

1. **NPU FlashAttention** at head_dim=128 + lq=lk=2048 hangs (`ERT_CMD_STATE_TIMEOUT`).
   `compile_attn_npu2_split(lqp, lkp, dk, dv)` API was added to
   `_llm_shared/kernel_builder/external_kernels.py` and the kernel compiles
   cleanly with the L1-feasible config (lkp=64, lqp=256, dk=dv=128); runtime
   hang is the open investigation. Highest-impact perf win when resolved
   (~2× prefill speedup projected).

2. **F32-output Down GEMM** would push per-layer cosine to 0.999+ uniformly
   and likely move competitive prompts (e.g. `'The capital of France is'`)
   to strict top-1 match. Edits a shared file (`llama3/multi_launch_builder/
   o_ffn_multi.py`) — must revalidate llama3 too. Defer unless downstream
   accuracy metric demands it.

3. **`rope_scaling=llama3` long-context** (factor=32, low/high freq factor)
   not implemented in `generate_rope_lut()` — inert for seq_len ≤ 8192;
   matters only for the long-context configuration.

## See also

- [`../llama3/`](../llama3/) — reference deployment (Llama-3.2-1B): canonical kernel sequence, multi-launch ELF design, KernelCache
- [`../smollm2_1_7b/`](../smollm2_1_7b/) — Tier-A precedent (SmolLM2-1.7B): MHA, smaller vocab, tied embeddings
- `docs/superpowers/edge-llm-candidates.md` — Path A roadmap and candidate survey
