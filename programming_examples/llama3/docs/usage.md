# LLAMA-3.2-1B Inference — Usage Guide

## Prerequisites

### Hardware & Toolchain
- AMD NPU2 hardware (Strix, AIE2P)
- MLIR-AIR installed with Peano compiler (`PEANO_INSTALL_DIR` set)
- Python environment with: `numpy`, `ml_dtypes`, `safetensors`, `transformers`, `huggingface_hub`, `filelock`

### Model Weights & Tokenizer (one-time setup)

The pipeline uses `meta-llama/Llama-3.2-1B` from HuggingFace. This is a gated model
that requires accepting Meta's license.

```bash
# 1. Install Python dependencies (if not already in your venv)
pip install safetensors huggingface_hub transformers ml_dtypes

# 2. Accept Meta's license agreement:
#    Go to https://huggingface.co/meta-llama/Llama-3.2-1B
#    Click "Accept" (requires a HuggingFace account, approval is usually instant)

# 3. Log in to HuggingFace CLI:
huggingface-cli login
#    Paste a token from https://huggingface.co/settings/tokens (needs "Read" permission)

# 4. First `make run` auto-downloads weights (~2.5GB) and tokenizer.
#    Cached at ~/.cache/huggingface/hub/ for future runs.
```

## Quick Start

```bash
cd programming_examples/llama3

# Step 1: Compile all kernels (one-time, ~4 minutes)
make compile

# Step 2: Run inference
make run
```

This runs the full pipeline: NPU prefill (processes prompt) → NPU decode (generates tokens).

---

## Commands

### `make compile`

Compiles all NPU kernels from source and caches them to disk.

- Compiles 6 external C++ kernels (rope, silu, attention, gemv) from `.cc` source
- Compiles 5 prefill ELF kernels via MLIR-AIR/aircc pipeline
- Compiles 3 decode ELF kernels via MLIR-AIR/aircc pipeline
- Results cached in `build_peano/prefill_kernel_cache/` and `build_peano/decode_kernel_cache/`
- Only needed once — subsequent runs use cached kernels via `--run-only`

### `make run`

Runs the unified inference pipeline with default settings (100 tokens).

```bash
make run                              # 100 tokens, default prompt (base model)
make run N_TOKENS=50                  # Generate 50 tokens
make run PROMPT="Once upon a time"    # Custom prompt
make run MODEL=instruct PROMPT="What is the capital of France?"  # Instruct model (Q&A)
```

**Model variants** (`MODEL=base` or `MODEL=instruct`):
- `base` (default): Text completion model. Continues the prompt.
- `instruct`: Instruction-following model. Answers questions using chat template.
  Same architecture and kernels — only the weights differ.
  Stops generation automatically on `<|eot_id|>` token.

What happens internally:
1. Loads model weights from HuggingFace cache
2. Pre-loads all weights into NPU buffer objects (one-time setup)
3. **NPU Prefill**: processes entire prompt through 16 transformer layers (~1.3s)
4. **NPU Decode**: generates tokens one at a time (~92ms each)

### `make profile`

Same as `make run` but prints per-token timing and kernel breakdown.

```bash
make profile
make profile N_TOKENS=10
```

Example output:
```
NPU prefill done in 1.54s. First token: 12366

Decoding 100 tokens (token 1 to 100)...
  Token 1: id=13, time=92ms
  Token 2: id=1102, time=91ms
  ...
  Token 100: id=578, time=92ms

Generated 100 tokens in 9.21s
Tokens/second: 10.86
Time/token: 92ms
```

### `make verify`

Runs inference and compares every intermediate result against a CPU F32 reference.
Useful for validating correctness after kernel changes.

```bash
make verify N_TOKENS=10
```

Checks:
- Per-layer KV cache correlation (NPU vs CPU)
- Logits correlation at prediction position
- Top-1 token match

### `make clean`

Removes all build artifacts (compiled kernels, `.o` files, temporary files).
Forces full recompilation on next `make compile`.

```bash
make clean
```

---

## Individual Pipelines

For development and debugging, prefill and decode can be run separately.

### Prefill Only

```bash
make compile-prefill                   # Compile prefill kernels
make run-prefill                       # Run 16-layer prefill with profiling
```

Runs `llama3_prefill.py`: processes a 2048-token prompt through all 16 layers
and outputs top-5 next-token predictions. Does not generate text.

### Decode Only

```bash
make compile-decode                    # Compile decode kernels
make run-decode                        # Run decode with CPU prefill for KV cache
```

Runs `llama3_decode.py`: uses CPU prefill to build the KV cache (slower, ~17s),
then runs NPU decode. Useful for testing decode kernels independently.

### CPU Reference

```bash
make run-reference                     # Run full model on CPU (F32, no NPU)
```

Runs `llama3_reference.py`: pure CPU F32 forward pass for verification baseline.

---

## Performance

| Phase | Time | Detail |
|-------|------|--------|
| **Prefill** | 1.30s kernel / 1.54s wall | 2048 tokens, 16 layers, NPU |
| **Decode** | 92ms/token | 10.8 tokens/sec, NPU |
| **100 tokens end-to-end** | ~11s | Prefill + decode |

Comparison with IRON:

| | AIR (this) | IRON | Speedup |
|---|---|---|---|
| Prefill | 1.30s (kernel) | 2.744s | 2.1x |
| Decode | 92ms/tok | 370ms/tok | 4.0x |

---

## File Structure

```
llama3/
├── Makefile                        ← Build commands (this guide)
├── llama3_inference.py             ← Unified pipeline: NPU prefill + NPU decode
├── llama3_prefill.py               ← Prefill-only pipeline
├── llama3_decode.py                ← Decode-only pipeline
├── llama3_weights.py               ← Weight loading from safetensors
├── llama3_reference.py             ← CPU F32 reference
│
├── kernel_builder/                 ← Shared kernel infrastructure
│   ├── stitching.py                ← MLIR text stitching for multi-launch ELFs
│   ├── gemm_builder.py             ← GEMM module builder + transform IR
│   ├── cache.py                    ← KernelCache, Profiler, prepare_air_project
│   └── external_kernels.py         ← C++ kernel compilation (rope, silu, attn, gemv)
│
├── multi_launch_builder/           ← Multi-launch ELF builders
│   ├── rms_gemms_rope_multi.py     ← Prefill: RMS+QKV+RoPE (6 launches)
│   ├── o_ffn_multi.py              ← Prefill: O+Add+FFN (8 launches)
│   ├── lm_head_multi.py            ← Prefill: LM Head (8 launches)
│   ├── rms_gemv_rope_multi.py      ← Decode: RMS+QKV+RoPE (6 launches)
│   ├── o_gemv_ffn_multi.py         ← Decode: O+FFN (8 launches)
│   ├── lm_head_gemv_multi.py       ← Decode: LM Head (8 launches)
│   ├── superseded/                 ← Previous builder versions (reference)
│   └── experimental/               ← Experimental builders (not in production)
│
├── ffn_swiglu/                     ← SiLU×mul kernel
├── build_peano/                    ← Build directory (created by make compile)
│   ├── prefill_kernel_cache/       ← Compiled prefill .elf files
│   ├── decode_kernel_cache/        ← Compiled decode .elf files
│   ├── *.o                         ← Compiled C++ kernels
│   └── air_project/                ← Temporary compilation artifacts
│
└── docs/                           ← Documentation
```

---

## Troubleshooting

**"Kernel not found in cache"**: Run `make compile` first, or remove `--run-only` flag.

**NPU lock timeout**: Another process is using the NPU. Check `lsof /tmp/npu.lock`.

**Slow first token**: The NPU enters power-save after ~10s idle. The warmup pass
handles this automatically. If running manually, ensure `prepare_runtime()` is called.

**Wrong results**: Run `make verify` to compare against CPU reference. Check that
`.o` files are fresh (`make clean` then `make compile`).
