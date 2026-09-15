# Reproduce the DFlash NPU study on another machine

This review branch packages the selected Qwen3-4B DFlash implementation and its
80-prompt performance study. It is a reproduction checkpoint, not an upstream
release. The selected B8 graph, kernels, and acceptance algorithm are preserved.

The historical development-machine result is **19.311 token/s B1 versus 34.999
token/s DFlash**, or **1.812x pooled speedup**. Mean request acceptance length is
4.579; accepted/proposed draft tokens is 47.53%. These numbers describe the
fixed cohort below, not a minimum speed for arbitrary prompts.

## Machine and software prerequisites

- Linux x86-64, tested on Ubuntu 24.04 with Python 3.12 and clang/lld 18.
- AMD NPU2 / AIE2P with the resources needed by the selected 30-compute-tile
  graph. Confirm the Kraken device and its driver expose this topology.
- Working AMD XDNA driver, firmware, and XRT NPU runtime with Python 3.12
  bindings. The original machine uses the XRT 2.21 family. Driver installation
  and firmware are machine-specific; this example does not replace them.
- Git, a C/C++ toolchain, Python venv support, and development dependencies for
  the normal MLIR-AIR build. See the repository's main installation guide.
- Internet access for compiler wheels, public model checkpoints, and the
  1.12-GB benchmark archive. Keep tens of GB free for compiler intermediates,
  models, and unpacked data. The original machine has 91 GiB usable RAM; live
  NPU prefill retains substantially more weights than decode-only benchmarking.

Check device visibility with `/opt/xilinx/xrt/bin/xrt-smi examine`. Keep other
NPU work idle during measurement. All Make build/run targets use
`/tmp/mlir-air-npu.lock`; a resident chat holds it until `/quit` or Ctrl-C.
Ctrl-Z suspends chat and retains the lock.

## Clone and build the AIR compiler

```bash
git clone --branch review/dflash-npu-20260915 --single-branch \
  https://github.com/tonyjie/mlir-air.git mlir-air-dflash-review
cd mlir-air-dflash-review
```

Use the commit/tag recorded in the release when you need an immutable checkout.
Do not substitute an arbitrary installed `air-opt`: this branch contains AIR
offset/ownership/scheduling fixes used by the selected graph.

If you already have the standard upstream environment, source it and build
only AIR from this checkout:

```bash
source /path/to/your/upstream/env.sh
python3.12 -m pip install -r programming_examples/llms/qwen3_4b_dflash/requirements-review.txt

flock --exclusive /tmp/mlir-air-npu.lock python3.12 \
  programming_examples/llms/qwen3_4b_dflash/setup_toolchain.py \
  --reuse-deps --llvm-link /path/to/llvm-21/bin/llvm-link --jobs 8
```

`--reuse-deps` requires `MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, and
`LLVM_INSTALL_DIR`. The script checks AIE and Peano versions before building.
It does not reuse your installed AIR compiler.

For a new Python environment and downloaded compiler dependencies:

```bash
flock --exclusive /tmp/mlir-air-npu.lock python3.12 \
  programming_examples/llms/qwen3_4b_dflash/setup_toolchain.py --jobs 8
```

This installs into `build_dflash_toolchain/`. Optional `--build-dir`,
`--xrt-dir`, `--llvm-link`, and `--cmake-modules` specify existing locations.
Without `--llvm-link`, it downloads the pinned LLVM 21 wheel and extracts that
one executable. The host LLVM 24 tools and Peano LLVM 22 remain separate.

The example only uses Transformers for tokenization; PyTorch is not required.
Transformers may print a notice about unavailable PyTorch models in a fresh
environment. The inference engines here are the NPU programs.

Both paths finish by writing a new environment file:

```bash
source build_dflash_toolchain/env.sh
cd programming_examples/llms/qwen3_4b_dflash
```

The exact versions and upstream base are in [toolchain.lock.json](../toolchain.lock.json).
As checked against upstream `46288593` on September 15, the dependency pins
match upstream. Our AIR base is `6746658f` plus the included fixes; the eight
newer upstream commits, including the fused-decode `air.api` migration, are not
part of this reproduction. See [toolchain.md](toolchain.md).

## Prepare models, kernels, and frozen inputs

```bash
make weights
make compile-study
make benchmark-data
make verify-host
make verify PREFILL=cpu
```

`make compile-study` builds the B8 target, B8 drafter, and the **B1 GEMV
performance baseline**, all at C4096 with live-context execution. B1 uses the
same settings as the historical study, including vector attention normalization
and no hidden-state export. B8 target/drafter each retain the selected checked
post-route assignments. Rebuild on the destination machine with the pinned
stack rather than substituting unrelated artifacts.

Both models use Q4NX packed weights, BF16 activations, and the original NPU
arithmetic. The official drafter is trained at block size 16; this delivery
executes B8 (one anchor plus seven proposals). Model and tokenizer revisions
are pinned in `model_paths.py`. `make weights` records source/cache checksums.
The paired study requires the measured packed-cache checksums. This was checked
by rebuilding both caches from the pinned model files. A stale cache from another
configuration is rejected before timing.

`make benchmark-data` downloads the release archive, checks its SHA-256, and
verifies every extracted prefill state. To transfer the archive manually:

```bash
make benchmark-data ARCHIVE=/path/to/dflash-benchmark-16x5-v1.tar.gz
```

The default data location is `build/benchmark_bundle/inputs`. Prompt text is
also readable in [benchmark/cases.json](../benchmark/cases.json):
`messages[].content` contains the task; `prompt` contains the exact Qwen chat
template; `input_ids` contains the actual tokens. Dataset provenance, selected
row indices, seed, and hashes are included. These are public benchmark tasks;
their upstream dataset licenses continue to apply.

## Run and compare

First run a short check: one prompt per dataset, capped at 256 output tokens.

```bash
make benchmark-study PER_DATASET=1 STUDY_TOKENS=256 \
  STUDY_DIR=build/study_smoke STUDY_LABEL=kraken
```

Then reproduce the full 80-prompt study with **one pass per mode**:

```bash
make benchmark-study STUDY_DIR=build/study_80 STUDY_LABEL=kraken
make study-report STUDY_DIR=build/study_80
```

Defaults are 16 prompts from each of GSM8K, MATH-500, HumanEval, MBPP, and
first-turn MT-Bench; a 2,048-token output cap; temperature zero; thinking
disabled; eight CPU workers; and C4096. Eight shards take two prompts from each
dataset and alternate B1/DFlash process order. Each process warms up before
timing. The original study spent about 47 minutes in timed decoding across both
modes, plus loading and warmup. The five-prompt smoke is not the full aggregate.

To resume an interrupted study, use the same command and add `RESUME=1`.
Source, input, artifact, weight, and option identities must match. Existing
studies are not overwritten. `STUDY_TOKENS` is separate from the live CLI's
`N_TOKENS`; `make benchmark` retains its single-mode interface.

The study records:

- `summary.json`: per-dataset and overall throughput, speedup, acceptance, and
  mean DFlash round/stage times, plus comparison with historical token outputs.
- `system.json`: CPU, affinity, memory, XRT platform, and runtime packages.
- `study.json`: exact source/input/weight/artifact identities and process order.
- `runs/`: per-request output tokens, acceptance traces, component timings, and
  process logs. Keep the whole study directory when sharing results.

A copied study can be summarized without the original machine paths. If needed,
set `REFERENCE=/path/to/benchmark_bundle/reference` on `make study-report`; the
default also looks for the locally downloaded reference bundle.

Record the machine's DDR data rate/channel configuration and power mode beside
these files if the platform report does not expose them. Higher DDR bandwidth
can improve B1 as well as DFlash; compare each machine's own B1/DFlash ratio.

## Timing and correctness boundaries

The performance study replays the **same frozen CPU-prefilled KV, hidden taps,
and first token** on each machine. Prefill is not repeated or timed. Decode time
excludes model loading, warmup, the first output token, and initial target KV
reset. It includes CPU context projection, transfers, NPU drafter/target calls,
and acceptance. This is the historical study's timing boundary.

Pooled throughput is total decoded tokens divided by total decode seconds.
Pooled speedup divides DFlash pooled throughput by B1 pooled throughput.
Equal-request throughput is the inverse of mean seconds/token across requests;
the macro equal-request speedup averages the five dataset ratios. These are
different weightings. B1 and B8 may generate different lengths, so speedup here
is a throughput ratio, not a fixed-output request-latency ratio.

`make verify` checks repeatability, committed KV, and exact output against
sequential row zero of the same B8 arithmetic. B1 GEMV is the performance
denominator, not a bit-exact oracle for B8 BFP16 matrix arithmetic. Historical
token/acceptance differences are recorded rather than hidden. The benchmark
does not grade answer accuracy or run generated code.

Live `make run`/`make chat` still default to NPU prefill. For those paths:

```bash
make compile-prefill
make verify
make chat N_TOKENS=256
```

See [validation.md](validation.md) for the NPU-prefill shared-exponent boundary
and the scope of exactness claims. The preserved reference results are in
`build/benchmark_bundle/reference`; [reference_summary.json](../benchmark/reference_summary.json)
is also tracked in this branch.
