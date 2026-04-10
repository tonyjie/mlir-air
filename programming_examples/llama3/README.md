# LLAMA-3.2-1B BF16 Inference on AMD NPU2

End-to-end LLAMA-3.2-1B (1B parameter, BF16) inference running on AMD NPU2 (AIE2P) hardware via MLIR-AIR. Supports both prefill (seq_len=2048) and autoregressive decode.

## Performance

| Phase | Time | vs IRON |
|-------|------|---------|
| Prefill (2048 tokens) | 1.30s kernel / 1.54s wall | 2.1x faster |
| Decode | 92ms/token (10.8 tok/s) | 4.0x faster |

## Quick Start

```bash
# One-time: compile all kernels (~4 min, cached to disk)
make compile

# Run inference (prefill + 100 tokens decode)
make run

# Run with custom prompt
make run PROMPT="In 1969, the first man to walk on"

# Run with profiling breakdown
make profile

# Run with correctness verification
make verify
```

## Documentation

| Doc | What's in it |
|-----|-------------|
| [Usage Guide](docs/usage.md) | All `make` targets, command-line options, file structure |
| [Performance Profile](docs/profile.md) | Kernel timing breakdown, BO categories, memory model |
| [Implementation Guide](docs/explain.md) | How kernels are built, compiled, and stitched together |
| [Known Issues](docs/issues.md) | BF16 precision, fixed seq_len, no sampling |

## Architecture Overview

Each transformer layer runs as 3 NPU invocations (prefill) or 3 invocations (decode):

```
Prefill (per layer):
  rms_gemms_rope  (6 launches) → flash_attn (1 launch) → o_ffn (8 launches)

Decode (per token, per layer):
  rms_gemv_rope   (6 launches) → CPU attention → o_gemv_ffn (8 launches)
```

Multiple operations are fused into single ELF binaries via multi-launch merging,
reducing XRT dispatch overhead from 10 calls/layer to 3.

## Key Files

| File | Purpose |
|------|---------|
| `llama3_inference.py` | Unified prefill + decode pipeline |
| `llama3_prefill.py` | Standalone prefill (with profiler report) |
| `llama3_decode.py` | Standalone decode |
| `llama3_weights.py` | Weight loading from HuggingFace safetensors |
| `llama3_reference.py` | CPU F32 reference implementation |
| `kernel_builder/` | Shared utilities: MLIR stitching, kernel cache, external kernel compilation |
| `multi_launch_builder/` | Multi-launch ELF builders (one per fused kernel) |
| `Makefile` | Build/run/profile/verify targets |

## Development History

For the optimization journey and design decisions, see [`docs/development_progress/`](docs/development_progress/):
- [Plan](docs/development_progress/plan.md) — original design and phased approach
- [Progress](docs/development_progress/progress.md) — milestone tracker
- [Performance History](docs/development_progress/perf_optimization.md) — 18.67s → 1.54s optimization journey
- [Multi-launch Merging](docs/development_progress/multi-launch/) — kernel fusion details
