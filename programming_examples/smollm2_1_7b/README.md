# SmolLM2-1.7B BF16 on NPU2 (MLIR-AIR)

End-to-end inference of [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
on AMD NPU2 (Strix, AIE2P) via MLIR-AIR. **Validated 2026-04-17** through the
7-phase `deploy-new-llm` skill chain — see
`docs/development_progress/phase6_finalize.md` for the full report.

## Performance

End-to-end NPU inference (`make run` / `smollm2_inference.py`):

| Stage | Latency | Per-unit | Notes |
|--|--|--|--|
| **NPU prefill** (24 layers, seq_len=2048) | **2.25 s** | 94 ms/layer | NPU FlashAttention + per-layer K/V extraction into KV cache |
| **First LM Head GEMV** (1 vector) | **17 ms** | — | Reuses the decode `lm_head_gemv` kernel |
| **NPU decode** (per token, 24 layers) | **137 ms** | 5.7 ms/layer | NPU rms_gemv_rope + CPU attn + NPU o_gemv_ffn + NPU LM Head GEMV |
| **Throughput (decode)** | **7.3 tok/s** | — | |

For Phase 4's standalone prefill perf (no K/V extraction overhead), prefill drops
to **1.88 s NPU / 79 ms-per-layer** (`make run-prefill`).

**Per-layer prefill and decode rates hit parity with Llama-3.2-1B** (79 vs 81 ms
prefill, 5.7 vs 5.75 ms decode), even though SmolLM2's MHA has 4× larger K/V
projections than Llama-3.2-1B's GQA. Total per-token rate is exactly proportional
to depth (24 vs 16 layers).

| Setup cost (one-time) | Time |
|--|--|
| HF weight download (first run) | ~30 s |
| Kernel compile — prefill | ~80 s (rms_gemms_rope + o_ffn) |
| Kernel compile — decode | ~22 s (rms_gemv_rope + o_gemv_ffn + lm_head_gemv) |
| Pre-load BOs — prefill | 2.83 s for 24 layers (3.1 GB) |
| Pre-load BOs — decode | 1.10 s for 24 layers + LM Head (3.6 GB) |

## Architecture vs. Llama-3.2-1B

SmolLM2-1.7B is structurally a Llama-family model — same RMSNorm, SwiGLU MLP,
half-split RoPE, BF16. The reference deployment is [`../llama3/`](../llama3/).
Divergences:

| Aspect | Llama-3.2-1B | SmolLM2-1.7B |
|--|--|--|
| Layers | 16 | **24** |
| n_heads / n_kv_heads | 32 / 8 (GQA) | 32 / 32 (**MHA**) |
| head_dim | 64 | 64 (same) |
| hidden_dim | 8192 | 8192 (same) |
| vocab_size | 128256 | **49152** |
| RoPE θ | 500000 | **130000** |
| Embedding tying | untied (lm_head separate) | **tied** (lm_head = embed_tokens) |
| Memory (BF16, weights only) | ~2.5 GB | ~3.4 GB |

**No new kernel code was needed.** Every divergence is handled by an existing
parametric path in the llama3 builders (`n_kv_heads` arg, tied-embedding fallback
in the loader, `n_partitions` arg in LM Head GEMV). See
`docs/development_progress/phase1_kernel_shapes.md` for the per-kernel breakdown.

## Quick start

### One-time setup
The first invocation will:
1. Download safetensors from HF to `~/.cache/huggingface/` (3.4 GB)
2. Compile external `.o` kernels into `build_peano/`
3. Compile multi-launch ELFs into `prefill_kernel_cache/` or `decode_kernel_cache/`

Subsequent runs reuse the cached artifacts — kernel compile is skipped (cache hit ~0.4 s).

### Make targets (recommended)

```bash
cd programming_examples/smollm2_1_7b

make help              # show all targets and options
make compile           # compile prefill + decode kernels (~2 min, one-time)
make run               # end-to-end NPU inference: prefill + decode (~3 s after one-time setup)
make profile           # prefill perf measurement (cold vs warm)
make verify            # decode with NPU/CPU top-1 verification (slow, 4 tokens)
```

`make run` invokes `smollm2_inference.py` — the unified script that does
**NPU prefill + KV-cache extraction + NPU LM Head + NPU decode** in one process.

Individual phase targets:
```bash
make run-reference     # Phase 0  CPU reference vs HuggingFace F32
make run-block         # Phase 2  single transformer block on NPU + NPU FlashAttention
make run-full          # Phase 3  full 24-layer NPU prefill + top-1 match (3 prompts)
make run-prefill       # Phase 4  prefill perf (cold vs warm with preload, no decode)
make run-decode-only   # Phase 5  CPU-prefill-seed + NPU decode (older path; for comparison)
```

Compile-only targets (useful in CI or to warm the cache without running inference):
```bash
make compile-prefill   # rms_gemms_rope + o_ffn + flash_attn (~80 s first time)
make compile-decode    # rms_gemv_rope + o_gemv_ffn + lm_head_gemv (~22 s first time)
make clean             # remove kernel caches and build artifacts
```

### Configurable options
Override via `make VAR=value`:

| Variable | Default | Purpose |
|--|--|--|
| `N_TOKENS` | 8 | Decode tokens to generate (Phase 5) |
| `PROMPT` | `"The capital of France is"` | Input prompt |
| `SEQ_LEN` | 2048 | Prefill sequence length |
| `MAX_SEQ` | 128 | Decode max position (KV cache size) |
| `MODEL` | `HuggingFaceTB/SmolLM2-1.7B` | HF model id or local snapshot path |

Examples:
```bash
make run N_TOKENS=20
make profile PROMPT='The meaning of life is'
make verify N_TOKENS=4              # CPU verify is slow (~25 s/token)
```

### Direct script invocation

The unified runner and each phase test can be invoked directly with finer-grained
flags (see `python3 <script>.py --help` for each):

```bash
python3 smollm2_inference.py --n-tokens 20 --profile      # end-to-end
python3 smollm2_reference.py --verify --prompt "..."      # Phase 0
python3 smollm2_phase2_test.py --npu-attn --no-preload    # Phase 2
python3 smollm2_phase3_test.py --diagnostic               # Phase 3 + per-layer drift
python3 smollm2_phase4_test.py --cpu-attn --n-warm-runs 5 # Phase 4
python3 smollm2_phase5_test.py --n-tokens 20 --max-seq 256 # Phase 5
```

### Sanity-check the install
```bash
python3 smollm2_weights.py        # ~10 s after download — no NPU activity
```

## File structure

| File | Purpose |
|--|--|
| `smollm2_inference.py` | **End-to-end NPU inference** — prefill (with K/V extraction) + decode |
| `smollm2_weights.py` | Config dataclass, HF safetensors loader, RoPE LUT generator |
| `smollm2_reference.py` | Pure-NumPy F32 reference forward pass (used as ground truth) |
| `smollm2_phase2_test.py` | Single-transformer-block NPU correctness |
| `smollm2_phase3_test.py` | Full 24-layer NPU prefill + CPU LM Head |
| `smollm2_phase4_test.py` | Prefill perf (cold vs warm with `preload_prefill_weights`) |
| `smollm2_phase5_test.py` | Decode loop with CPU-prefill-seed + NPU LM Head GEMV |
| `llama3_*.py` | Inherited orchestration helpers (config-driven; reused as-is per LESSONS.md Lesson 1 — not renamed to avoid cross-file import friction) |
| `multi_launch_builder/` | Inherited multi-launch ELF builders (config-driven) |
| `CLAUDE.md` | Model-specific guide (divergences from llama3, file conventions) |
| `TODO.md` | Phase status (all 7 PASSED) |

## Documentation

| Doc | Content |
|--|--|
| [`TODO.md`](TODO.md) | Phase checkboxes and one-line summaries |
| [`CLAUDE.md`](CLAUDE.md) | Model-specific guide for future Claude sessions |
| [`docs/development_progress/progress.md`](docs/development_progress/progress.md) | Full phase-by-phase log |
| [`docs/development_progress/phase6_finalize.md`](docs/development_progress/phase6_finalize.md) | End-to-end perf summary + reusable-pattern audit |
| [`docs/development_progress/phase1_kernel_shapes.md`](docs/development_progress/phase1_kernel_shapes.md) | Per-kernel shape inventory (drop-in vs recompile) |
| [`docs/development_progress/phase3_full.md`](docs/development_progress/phase3_full.md) | Per-layer cosine-sim drift across 24 layers |
| [`docs/development_progress/phase4_prefill.md`](docs/development_progress/phase4_prefill.md) | Prefill perf with the 5 optimization patterns |
| [`docs/development_progress/phase5_decode.md`](docs/development_progress/phase5_decode.md) | Decode pipeline + LM Head GEMV details |
| [`docs/development_progress/LESSONS.md`](docs/development_progress/LESSONS.md) | 4 captured lessons + skill-update recommendations |
| [`../llama3/CLAUDE.md`](../llama3/CLAUDE.md) | Canonical kernel sequence, multi-launch ELF design, BO pre-loading patterns |

## Open follow-ups

| Item | Type | Estimated effort |
|--|--|--|
| LM Head GEMV right-sizing (`n_partitions=3` exact for vocab=49152) | Perf, ~3 ms/token (~2%) | 1-2 h |
| Reduce per-layer K/V-extraction overhead in NPU prefill (currently ~15 ms/layer) | Perf, ~0.4 s on prefill | 1-2 h |
| Cleanup: drop the inherited `llama3_*.py` files now that tests work via `../llama3/` imports | Hygiene | 30 min |

See [`docs/development_progress/phase6_finalize.md`](docs/development_progress/phase6_finalize.md)
for the full follow-up list.
