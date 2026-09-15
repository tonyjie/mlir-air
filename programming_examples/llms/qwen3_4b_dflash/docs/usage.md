# Usage

Run commands from `programming_examples/llms/qwen3_4b_dflash`.

## Environment

The standard MLIR-AIR setup defines `MLIR_AIR_INSTALL_DIR`,
`MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, and `LLVM_INSTALL_DIR`:

```bash
source ../../../utils/env_setup.sh \
  /path/to/air-install /path/to/mlir-aie-install \
  /path/to/peano-install /path/to/llvm-install
```

`env.sh` validates these variables, adds the tools and Python bindings to the
runtime paths, clears model-builder variables left by other examples, and fixes
CPU worker counts at eight. Set `DFLASH_AIR_INSTALL` to override only the AIR
installation. Set `DFLASH_BASE_ENV` to an existing environment script if that is
how the local build is normally entered. Set `DFLASH_LLVM_LINK` to an LLVM <23
`llvm-link`; only that executable is shimmed ahead of Peano. See
[Toolchain](toolchain.md).

Install the additional Python dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

## Weights

```bash
make weights
```

The default target source is `FastFlowLM/Qwen3-4B-NPU2`. The official draft is
`z-lab/Qwen3-4B-DFlash-b16` at revision
`b74e3a329c4d963783143b1e970d95b002be72bd`. Missing source files are downloaded
through Hugging Face. Packed caches default to:

```text
~/.cache/q4nx_qwen3_4b/requant_w2ch.npz
~/.cache/qwen3_4b_dflash_npu/draft_requant_w2ch.npz
```

Override them with `TARGET_MODEL`, `TARGET_CACHE`, `DRAFT_CACHE`, and
`DRAFT_SNAPSHOT` Make variables or the corresponding `DFLASH_*` environment
variables. `prepare_weights.py --offline` validates/builds from an already
cached snapshot without network access.

## Build

```bash
make compile
```

This produces raw target and drafter builds, applies the checked route changes,
and leaves runnable artifacts in:

```text
build/target/
build/draft/
build/prefill_C2048/
```

The default also builds the eight standard Q4NX NPU prefill kernels. Existing
users can build just this addition with `make compile-prefill`. The shared
kernel cache checks its toolchain identity and rebuilds when that changes.
The adapter also checks `dflash_prefill_config.json` so a cache compiled for a
different sequence length cannot be reused silently.

The Makefile fixes every selected decode graph option. C4096 is intentional and cannot
be overridden from Make. Builds and device commands wait on
`/tmp/mlir-air-npu.lock`.

For a custom build directory:

```bash
make compile BUILD_DIR=/path/to/build
```

## Run and verify

```bash
make run PROMPT="What is the capital of France?" N_TOKENS=64
make chat N_TOKENS=256
make verify-host
make verify
```

`make run` streams by default. The prefill token appears first; each later
fragment contains only the accepted prefix and the verifier's next token.
DFlash emits up to eight tokens per round, so text arrives in short bursts at
the actual round cadence. Partial UTF-8 characters are held until complete.
Statistics print after the response, and the full text/token IDs are saved to
`build/last_run.json`. Set `STREAM=0` to print the response after decoding:

```bash
make run PROMPT="Explain speculative decoding." N_TOKENS=128 STREAM=0
```

`make run` and `make chat` default to `PREFILL=npu`. This uses the standard
`qwen3_4b_q4nx` prefill kernels, with copies of layer outputs 2, 10, 18, 26, and 34
for DFlash. Q4NX weights are dequantized on the host once during initialization
and stored as resident BF16 prefill weights. Transformer layers and the LM head
run on the NPU; embedding lookup, final normalization of one row, and state
packaging remain on the host, matching the reference example.

`make chat` prepares the tokenizer, speculative pair, context projection, and
NPU prefill before `Prompt>`, and retains them for subsequent requests. Each
submitted prompt performs prefill, KV reset, and streaming generation. The
original CPU reference remains available explicitly:

```bash
make run PREFILL=cpu PROMPT="What is the capital of France?"
make chat PREFILL=cpu
```

CPU chat retains about 8.22 GiB of decoded prefill weights. NPU prefill also
keeps decoded weights plus resident device buffers, so its memory footprint is
larger than the decode-only runtime.

The standard prefill has a fixed padded shape. `PREFILL_CTX=2048` supports up to
2,048 real prompt tokens; shorter prompts still use the padded kernel. The
shared GEMM registry also supports `PREFILL_CTX=4096`, which creates a separate
cache and takes longer per prefill. The combined prompt/output/final B8 block
must still fit the decode capacity C4096. The delivery validation uses the
2,048-row prefill. For the longer shape:

```bash
make compile-prefill PREFILL_CTX=4096
make chat PREFILL_CTX=4096
```

Each prompt is independent, matching the Llama example: previous responses are
not added to the context. Blank input is ignored. `/quit`, `/exit`, Ctrl-D, or
Ctrl-C exit the session. Overlong requests are rejected before prefill and the
prompt loop continues. `N_TOKENS` is the output limit per request, including
the first prefill token and any EOS token. Thinking remains disabled.

Completed chat requests are appended to `build/chat.jsonl`; override this with
`CHAT_LOG=/path/to/session.jsonl`. The NPU lock is held for the entire chat
session, including time waiting for input. Exit the session to release it.

Timing fields have separate boundaries:

- Initialization is printed once and excludes user input and prompt prefill.
- `prefill_backend` identifies `npu` or `cpu`; `prefill_seconds` includes that
  prompt forward pass and KV/tap packaging. NPU weight preparation is part of
  initialization. `npu_prefill_seconds` or `cpu_prefill_seconds` records the
  applicable backend; the other field is null.
- `time_to_first_token_seconds` starts at submission and covers tokenization,
  prefill, and initial KV reset. It excludes model initialization and user
  typing time.
- `decode_tps` retains the benchmark convention: tokens after the first prefill
  token divided by DFlash loop wall time. Streaming decode includes the small
  text-decoding/printing callback cost; `make benchmark` remains non-streaming.
- `request_seconds` in the saved result covers prompt processing and generation,
  excluding initialization, user input, final statistics, and result-file writes.

`make verify-host` needs no NPU and exhaustively checks prefix acceptance,
bonus-token, clipping, and EOS behavior, plus offline UTF-8 streaming checks.
`make verify` checks prefill repeatability and prefix behavior, then compares
DFlash with sequential row-zero execution of the same B8 target arithmetic
from that identical prefill state, including committed target KV. See
[Validation](validation.md) for the shared prefill's BFP16 numerical boundary.
`make verify PREFILL=cpu` retains the original CPU-prefill gate.

Direct invocation exposes cache and artifact paths:

```bash
flock --exclusive /tmp/mlir-air-npu.lock ./env.sh python3 run.py \
  --target-artifacts build/target --draft-artifacts build/draft \
  --target-cache /path/to/target.npz --draft-cache /path/to/draft.npz \
  --prompt "Explain speculative decoding." --tokens 64

# The Make targets take the device lock automatically; do the same directly.
flock --exclusive /tmp/mlir-air-npu.lock ./env.sh python3 run.py \
  --interactive --tokens 256 --output build/chat.jsonl
```

## Saved-input benchmark

The published 35.00 token/s is pooled decode throughput over 80 fixed prompts,
with 16 seed-0 examples each from GSM8K, MATH-500, HumanEval, MBPP, and first-turn
MT-Bench. It is not a minimum speed for an arbitrary prompt. Acceptance and
request length matter: the measured dataset averages ranged from 20.96 token/s
for MT-Bench to 43.33 token/s for MATH-500. The earlier 39.19 token/s result used
a smaller, different 20-prompt cohort.

On this development machine, all 80 original prompt states are already prepared.
To run one DFlash pass with the same cohort and 2,048-token cap:

```bash
make benchmark \
  INPUTS=/home/jiajli/apps/dflash/results/npu/benchmark_validity_16x5_20260914/inputs \
  N_TOKENS=2048 PER_DATASET=16 MODE=spec \
  OUTPUT=build/benchmark_80_spec.jsonl
```

This prints per-request results as they finish and a per-dataset/overall summary
at the end. Pooled throughput is the sum of decoded tokens divided by the sum
of decode seconds; it is not the arithmetic average of request throughputs.
The earlier DFlash pass spent about 17 minutes in timed decoding. Allow extra
time for loading and warmup; timings vary with current system conditions.

A quicker pipeline check uses one example per dataset and a shorter output cap:

```bash
make benchmark \
  INPUTS=/home/jiajli/apps/dflash/results/npu/benchmark_validity_16x5_20260914/inputs \
  N_TOKENS=256 PER_DATASET=1 OUTPUT=build/benchmark_smoke.jsonl
```

That shorter run does not reproduce the published aggregate. Output files are
not overwritten. Set `RESUME=1` with the same options and output path to resume
an interrupted run. Show the summary again without using the NPU:

```bash
make benchmark-report OUTPUT=build/benchmark_80_spec.jsonl
```

The benchmark replays **frozen CPU-prefilled KV, hidden taps, and first tokens**
for comparison with the original study. `PREFILL=npu` controls live `run`, `chat`,
and `verify`; it does not change these saved benchmark inputs. Decode timing
excludes initialization, warmup, prefill, the first output token, and initial KV
reset. It includes the CPU context projection, NPU draft/verify calls, transfers,
and acceptance in every DFlash round. This command reproduces the DFlash side
of the study; it does not rerun the alternating B1/DFlash process schedule.

For portable input preparation, `prepare_inputs.py` converts a frozen JSON
prompt/token manifest into the same cache format:

```bash
./env.sh python3 prepare_inputs.py \
  --cases /path/to/source_cases.json --output build/benchmark_inputs \
  --per-dataset 16

make benchmark INPUTS=build/benchmark_inputs MODE=spec \
  OUTPUT=build/benchmark_spec.jsonl N_TOKENS=2048

make benchmark INPUTS=build/benchmark_inputs MODE=control \
  OUTPUT=build/benchmark_control.jsonl N_TOKENS=2048
```

The `control` mode uses sequential row zero of the same B8 target arithmetic and
is the exact correctness oracle. It is not the B1 performance denominator.
Building the separate B1 GEMV performance denominator is outside this minimal
selected-pair target.

## Cleanup

```bash
make clean
```

This removes only `build/`. Hugging Face downloads and packed caches are retained.
