# Qwen3-4B DFlash B8 on AMD NPU2

This directory is the minimal delivery of the selected NPU DFlash
implementation. It runs the official `Qwen/Qwen3-4B` target with
`z-lab/Qwen3-4B-DFlash-b16`, using eight-token drafting and verification on
AMD NPU2 through MLIR-AIR.

The delivered configuration is fixed to the validated implementation at source
checkpoint `3d7bfc3d3b5e6f19f0ccb195edb2dc0ae0ad407c`. Experimental projection,
layout, B16, and timing-diagnostic variants are intentionally excluded.

## Measured result

On the Ryzen AI development system, the selected pair was measured on 80 fixed
prompts—16 each from GSM8K, MATH-500, HumanEval, MBPP, and first-turn MT-Bench:

| Mode | Pooled decode throughput |
|---|---:|
| NPU target-only B1 | 19.31 token/s |
| NPU DFlash B8 | 35.00 token/s |
| Speedup | 1.81x |

The prompt-bootstrap 95% interval for pooled speedup is 1.70–1.94x. Mean
acceptance length is 4.58 when every request has equal weight. See
[Validation and measured scope](docs/validation.md) before comparing this number
with the paper or another backend.

## Prerequisites

1. An MLIR-AIR checkout built for NPU2, including the compiler fixes present in
   this branch, plus compatible MLIR-AIE, Peano, and XRT installations.
2. Python packages from `requirements.txt` in the active MLIR-AIR environment.
3. Access to the public Hugging Face target and draft checkpoints. The first
   `make weights` downloads missing source weights and creates local packed
   caches; approximately 3.2 GB is required for the two packed caches.

Set up the standard MLIR-AIR environment before invoking the Makefile:

```bash
source ../../../utils/env_setup.sh \
  /path/to/air-install \
  /path/to/mlir-aie-install \
  /path/to/peano-install \
  /path/to/llvm-install
```

Alternatively, `env.sh` can source an existing environment file:

```bash
export DFLASH_BASE_ENV=/path/to/mlir-air-build/env.sh
export DFLASH_AIR_INSTALL=/path/to/air-install
```

All build and device targets take `/tmp/mlir-air-npu.lock` automatically.

## Quick start

```bash
python3 -m pip install -r requirements.txt

# One-time model preparation.
make weights

# One-time target/drafter compile, checked routing, and NPU prefill compile.
make compile

# Pure host acceptance-rule check, then NPU end-to-end verification.
make verify-host
make verify

# Greedy, thinking-disabled generation with streaming output.
make run PROMPT="Explain speculative decoding." N_TOKENS=64

# Load once, then enter independent prompts interactively.
make chat N_TOKENS=256
```

`make run` streams committed text after each DFlash verification round and
prints timing/acceptance statistics afterward. `make chat` prepares the models
and resident NPU prefill weights before displaying `Prompt>` and reuses them across
requests. Each prompt starts a fresh context. See [Usage](docs/usage.md) for
timing boundaries, logs, and controls.

Compilation produces `build/prefill_C2048` for the standard Q4NX NPU prefill
and `build/target` and `build/draft` for speculative decoding. Each decode directory contains
one C4096 dynamic-context xclbin, its transaction template, and a manifest that
binds configuration, source, compiler, and binary hashes. Runtime code rejects
artifacts whose ABI, configuration, or hashes do not match.

## What runs where

```text
NPU prompt prefill (standard Q4NX kernels, five captured layer outputs)
  -> initial target KV + five hidden-state taps + first token
  -> CPU learned context projection and drafter-KV injection
  -> NPU drafter B8 dispatch (anchor + seven masks)
  -> CPU argmax proposals
  -> NPU target B8 dispatch (causal verification + hidden taps)
  -> CPU greedy prefix acceptance and commit
  -> repeat
```

The target and drafter are separate NPU programs and separate xclbins. Each
speculative round dispatches the drafter once and the target once. The CPU owns
the committed position and acceptance decision; rejected KV slots are
overwritten on the next round.

Prefill defaults to the existing Qwen3-4B Q4NX NPU path: host weight
preparation/embedding lookup, NPU transformer layers and LM head, and a small
host final norm. Set `PREFILL=cpu` to use the original CPU reference. The
prefill kernels process a padded 2,048-row prompt; the B8 decode capacity remains
4,096. See [Usage](docs/usage.md) for longer prompts and timing boundaries.

The measured decode throughput depends on the prompt and acceptance rate. To
reproduce the fixed 80-prompt study, use the [benchmark command](docs/usage.md#saved-input-benchmark),
which replays the original saved prompt states with a 2,048-token output cap.

## Documentation

| Document | Purpose |
|---|---|
| [Architecture](ARCHITECTURE.md) | Model geometry, NPU map, round dataflow, KV injection |
| [Usage](docs/usage.md) | Environment, weights, build, run, verify, benchmark, cleanup |
| [Validation](docs/validation.md) | Correctness boundary, measured performance, known limitations |
| [Toolchain](docs/toolchain.md) | Compatible LLVM linker, known-good stack, fresh-build evidence |

## Source map

| File | Purpose |
|---|---|
| `selected_config.py` | Single supported target/drafter configuration |
| `target_config.py`, `batch_graph.py`, `fused_attention_graph.py` | AIR module and buffer ABI |
| `kernels/` | Selected projection, normalization, GLU, RoPE-adjacent and parallel-attention kernels |
| `build_target.py`, `route_artifact.py` | Compile and checked post-route packaging |
| `target_runtime.py` | Manifest-driven XRT execution and KV/state access |
| `draft_weights.py`, `packed_weights.py`, `prepare_weights.py` | Official draft adapter and packed-cache preparation |
| `context_prepass.py` | CPU target-feature fusion and learned context K/V projection |
| `speculative.py` | Greedy DFlash acceptance, commit, EOS, and token-budget logic |
| `npu_prefill.py` | Adapter over standard Q4NX NPU prefill with five hidden-state taps |
| `runtime.py`, `run.py`, `streaming.py` | Inference API, streaming CLI, and resident chat |
| `verify.py`, `test_streaming.py` | Host acceptance/streaming checks and NPU repeatability/same-B8 oracle gate |
| `prepare_inputs.py`, `run_benchmark.py`, `benchmark_summary.py` | Frozen-input benchmark and pooled per-dataset summary |
