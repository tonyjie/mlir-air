# Add SmolVLA, with its SigLIP vision encoder running on NPU2

Adds [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — a
Vision-Language-Action robot policy — as `programming_examples/llms/smolvla/`,
with its vision encoder and connector running on NPU2 via MLIR-AIR and spliced
into the unmodified LeRobot pipeline. First non-LLM model in `llms/`, built on
the existing `shared/` infrastructure and kernel registry.

## Result

| | Action chunk | |
|---|---|---|
| Pure CPU (unmodified lerobot) | 913 ms | 1.00× |
| **NPU vision + connector** | **851 ms** | **1.07×** |

`make profile` reproduces this — one process, both arms warmed, interleaved,
median of 10 — at 925.9 ms against 856.5 ms (**1.081×**), with the vision stage
itself 550.7 → 462.5 ms (**1.19×**). Eight quantities agree within a few percent
across two unrelated harnesses.

Correctness gate (`make verify`), against the unmodified CPU model with noise
pinned:

```
cosine 0.998996   (gate >= 0.99)
nMSE   0.003023   (gate <= 0.04)
[verify] PASS
```

All measurements on AMD Ryzen AI 9 HX 370 / NPU2 (Strix, AIE2P), CPU governor
and EPP `performance`, NPU `pmode=Turbo`.

## Why only the vision stage

SmolVLA is three stages. **All three were ported to the NPU and verified. Only
the vision encoder ships there**, because it is the only one measurably faster.

| Stage | Shape · frequency | CPU | NPU | Ships on |
|---|---|---|---|---|
| SigLIP vision + connector | seq **1024** · ×3 cameras | 546 ms | **465 ms** | **NPU, 1.19×** |
| SmolLM2-360M backbone | seq 256 · ×1 | **77 ms** | 229 ms | CPU, NPU ~3× slower |
| Action expert | seq 50 · **×10 denoise steps** | **285 ms** | ~4× CPU | CPU, NPU ~4× slower |

The NPU wins when the shapes fill its 8×4 array and loses when they do not.
Measured GFLOP/s for the *same* matmul kernels: seq 1024 → 3798–5790,
seq 256 → 2046–3366, seq 50 → 25–1163. The expert's 50 tokens pad to 64, which
admits only a tiling occupying half the array. Two further reasons: every launch
costs ~85 µs regardless of the work in it, and the registry FlashAttention
kernel applies no mask, so the two smaller stages fall back to attention
decomposed into 11 dispatches per layer instead of 1.

**The backbone and action-expert NPU ports are not in this PR.** They exist and
pass their gates; including them would roughly double the diff. They are on the
`smolvla` branch and can land separately.

Vision is 54% of the NPU-path wall clock, so even an infinitely fast vision
stage caps the end-to-end gain at ~1.85×; the CPU backbone and expert are the
ceiling. Within vision, 91% is device time — fusion has squeezed host glue to
40 ms, so further gains have to come from the kernels.

## What is in the diff

50 files.

| Path | What |
|---|---|
| `llms/smolvla/` | the example — 8 Python files (~2,600 lines), Makefile, README, ARCHITECTURE, 5 docs, 2 lit tests |
| `kernel_registry/` | 19 new GEMM shapes, new **LayerNorm** and **GELU** detail pages, a per-shape `herd` override |
| `llms/shared/` | new `infra/thread_limits.py`; `o_ffn_multi.py` made registry-driven per shape |
| `llms/verify/` | `regression_gate` for continuous-output models |
| `programming_examples/lit.cfg.py` | new `lerobot` feature (additive) |
| `.github/workflows/` | install this example's requirements — see below |

Layout mirrors `llama32_1b/` and `qwen3_1_7b/`. The `smolvla_vision_*` trio is
the whole NPU-mapped stage; adding another stage later means
`smolvla_backbone_*` beside it, so the file names say which stages are mapped.

## Three things a reviewer should look at

### 1. This example needs numpy ≥ 2, which conflicts with mlir-aie's pin

lerobot requires `numpy>=2.0,<2.3`. `mlir-aie/python/requirements.txt` pins
`numpy>=1.19.5, <2.0` with the comment *"2.1 would be nice … but it doesn't
seem to work well with pyxrt"*. The CI change in this PR installs this
example's requirements after that, so pip resolves numpy to 2.x for the whole
job.

That was measured before enabling, not assumed. Under numpy 2.2.6 on NPU2:

| Suite | Total | Pass | Fail |
|---|---|---|---|
| `python/test` (Python bindings) | 21 | **21** | 0 |
| `test` (core + xrt e2e) | 214 | 193 | 21 |
| `programming_examples` (non-LLM) | 322 | **213** | 1 |
| LLM `make verify` (smolvla, qwen25_0_5b, qwen3_1_7b) | 3 | **3** | 0 |

None of the 22 failures involve numpy: 21 are link errors for
`libxaiengine`/`libairhost`/`libsysfs` (this install does not build the airhost
runtime) and 1 is an aiecc MLIR lowering error on `vector.contract`. The 213
passing programming examples are the ones that actually dispatch through pyxrt.
`qwen25_0_5b` and `qwen3_1_7b` were also run *before* the upgrade and pass
identically.

**Caveat:** the same suites were not re-run under numpy 1.x for a controlled
comparison, so strictly this shows the 22 failures exist under numpy 2, not
that they are pre-existing. Both failure modes (missing `.so`, MLIR
legalization) are outside any numpy code path.

If maintainers would rather not move numpy, **drop the one line in
`.github/workflows/buildAndTestRyzenAI.yml`**: `run_npu2_verify.lit` then
reports UNSUPPORTED instead of failing, and the compile test still runs.

### 2. The registry carries rows for code this PR does not ship

15 of the 19 new GEMM shapes are action-expert shapes. They are real
measurements on real hardware, and the registry's purpose is to record every
verified (kernel, shape) so future ports can look them up. Dropping them would
only mean re-measuring later.

### 3. The gate's headroom is smaller than the passing numbers suggest

The gate runs on **all-zero images, all-zero state and zero noise**. Pinning
the noise is necessary for reproducibility; the zero images are a real limit.

| Input | cosine | nMSE | headroom |
|---|---|---|---|
| all-zero (what the gate runs) | 0.998996 | 0.003023 | cos 10× / nMSE 13× |
| fixed-seed random images + state | 0.996201 | 0.009976 | **cos 2.6× / nMSE 4×** |

Judge by the second row. A consequence worth knowing: three all-zero images
produce three identical encodings, so the gate cannot detect per-camera
misordering — the assertions in `run_hybrid_forward` cover a case the gate
structurally cannot.

## Known issue, not fixed here

`compile_gemm_mm` bakes `DIM_M=tile_m`, `DIM_N=tile_n` and `DIM_K=tile_k_l1`
into the external microkernel, but the shared helper names the object from
`tile_n` alone. Two GEMMs needing different reduction chunks can therefore
claim one filename, and since `compile_gemm_mm` writes with `force=True`, the
loser silently links a microkernel built for the other's `DIM_K`. Symbols
match, so nothing errors.

No currently shipped model is known to be wrong: they all use `tile_k_l1=32`
with `tile_m` at the method default. **This PR does raise the exposure**,
because several registry rows it adds specify `tile_m=16` and `tile_k_l1=48`.

I attempted a fix and reverted it: it broke Qwen2.5-0.5B (`make verify` PASS
2/2 → FAIL 0/2) while all ten models still compiled clean. The likely cause is
that `gemm_method_spec()` still returns the method-keyed name — **a hypothesis,
not a verified finding.** Doing it properly needs three call sites made
consistent at once, a recorded per-model `make verify` baseline taken *before*
any edit, and **`make verify` rather than `make compile` as the gate** —
compiling only proves the linker found an object, not that its symbols mean
what the IR expects.

## Notes

**SmolVLA does not use the token-set gate.** It emits a continuous action
chunk, so `verify_adapter.py` applies `regression_gate`. The reference is the
unmodified lerobot model computed live in the same process — not a saved
fixture, which would go stale and which made the verify lit test fail on a
clean checkout.

**Both lit tests were executed**, not just inspected:

```
PASS: llms/smolvla/run_npu2_compile.lit   (79.13s)
PASS: llms/smolvla/run_npu2_verify.lit    (13.65s)
```

and with lerobot absent, compile still passes while verify reports UNSUPPORTED.
Compile needs no torch/lerobot at all — checked by making them unimportable and
confirming all 5 ELFs still build.

**Two host operations stay on the CPU deliberately**: the im2col patch
embedding (a one-time reshape before the layer loop) and the connector's
pixel-shuffle (a pure space-to-depth reshape with zero arithmetic, verified
bit-exact against HuggingFace). The connector's actual math, a 64×12288×960
projection, runs on the NPU.

**The scoped BLAS thread clamp currently measures as a no-op here.** The code
it came from cites 135 → 178 ms per image; an A/B on `SMOLVLA_NPU_BLAS_LIMIT`,
three runs each, gives 442.5 ms clamped and 441.1 ms unclamped. It is kept for
machines where the contention is real, and `docs/explain.md` says so.

## Reproduce

```bash
cd programming_examples/llms/smolvla
pip install -r requirements.txt   # torch + lerobot[dataset] + num2words
make compile        # build every vision ELF; no NPU dispatch, no download
make verify         # the gate
make profile        # CPU vs NPU, interleaved, with the per-ELF breakdown
make cpu-baseline   # run the unmodified CPU model on its own
```

On a shared machine, wrap device-touching targets the way the siblings are
wrapped: `flock -x -w 1800 /tmp/mlir-air-npu.lock make verify`.
