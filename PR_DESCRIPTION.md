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

Vision alone is **1.19× per image** (155 vs 184 ms). Correctness gate
(`make verify`), against the unmodified CPU model with noise pinned:

```
cosine 0.998996   (gate >= 0.99)
nMSE   0.003023   (gate <= 0.04)
[verify] PASS
```

All measurements on AMD Ryzen AI 9 HX 370 / NPU2 (Strix, AIE2P), CPU governor
and EPP `performance`, NPU `pmode=Turbo`, machine idle, configurations
interleaved.

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
pass their gates, but including them would roughly double the diff and pull in a
new `masked_softmax` example that only they need. They can land separately.

## What is in the diff

| Path | What |
|---|---|
| `llms/smolvla/` | the example — 22 files, ~2,900 lines of Python |
| `kernel_registry/` | 19 new GEMM shapes, new **LayerNorm** and **GELU** detail pages, a per-shape `herd` override in `registry_lookup.py` |
| `llms/verify/` | `regression_gate` for continuous-output models |
| `llms/shared/` | `o_ffn_multi.py` made registry-driven per shape rather than assuming fused-cast |

Layout mirrors `llama32_1b/` and `qwen3_1_7b/`: `<model>_*.py`, `Makefile`,
`README.md`, `ARCHITECTURE.md`, `requirements.txt`, `docs/{explain,profile,usage}.md`,
`docs/detail/`, and two `.lit` files.

## Notes for the reviewer

**The registry carries rows for code this PR does not ship.** 15 of the 19 new
GEMM shapes are action-expert shapes. They are real measurements on real
hardware, and the registry's purpose is to record every verified (kernel, shape)
so future ports can look them up. Dropping them would only mean re-measuring
later.

**SmolVLA does not use the token-set gate.** It emits a continuous action chunk,
not tokens, so `verify_adapter.py` applies `regression_gate` rather than
`compute_topk_set_check`. The reference is the unmodified lerobot model on CPU,
not a reimplementation, and the comparison is on the tensor the robot would
actually execute.

**The `.lit` files have not been executed.** They are modelled line for line on
`qwen3_1_7b`'s, and both `CHECK` strings were confirmed present in the
corresponding `make` target output (`Compilation passed.`, `[verify] PASS`), but
the suites were not run — the configured lit tree points at the main checkout
rather than the branch. Worth running once in CI.

**Two host operations stay on the CPU deliberately**, and are not fallbacks from
a broken NPU path: the im2col patch embedding (a one-time reshape before the
layer loop, not hot-loop work) and the connector's pixel-shuffle (a pure
space-to-depth reshape with zero arithmetic, verified bit-exact against
HuggingFace). The connector's actual math, a 64×12288×960 projection, runs on
the NPU.

## Known issue, not fixed here

`compile_gemm_mm` bakes three compile-time macros into the external microkernel
— `DIM_M=tile_m`, `DIM_N=tile_n` and `DIM_K=tile_k_l1` — but the object is named
from the *method* alone (`_m32` for drain, `_m64` for fused-cast), which encodes
a proxy for `tile_m` and nothing else. Two GEMMs needing different reduction
chunks can therefore claim one filename, and since `compile_gemm_mm` writes with
`force=True`, the loser silently links a microkernel built for the other's
`DIM_K`. Symbols match, so nothing errors.

No currently shipped model is known to be wrong: they all use `tile_k_l1=32`
with `tile_m` at the method default, so the omitted dimensions happen to be
constants.

**This PR does raise the exposure**, because several of the registry rows it
adds specify `tile_m=16` and `tile_k_l1=48`. Anyone looking those shapes up to
build a fused ELF would get an object named from the wrong dimensions.

I attempted a fix in this branch and reverted it: it changed the naming in
`gemm_registry_config` only, and broke Qwen2.5-0.5B (`make verify` PASS 2/2 →
FAIL 0/2) while all ten models still compiled clean. The likely cause is that
`gemm_method_spec()` still returns the method-keyed name, so IR built through
both paths disagrees on the symbol suffix — **a hypothesis, not a verified
finding.**

Doing it properly needs three call sites made consistent at once
(`_spec_with_tiles`, `gemm_method_spec`'s direct callers, and the hardcoded
`compile_gemm_mm` calls in ten models), a recorded per-model `make verify`
baseline taken *before* any edit, and **`make verify` rather than `make compile`
as the gate** — compiling only proves the linker found an object, not that its
symbols mean what the IR expects. That is its own change with its own
verification matrix; this PR does not depend on it.

## Reproduce

```bash
cd programming_examples/llms/smolvla
make compile   # build every vision ELF; no NPU dispatch, no download
make oracle    # regenerate the CPU baseline (CPU only)
make verify    # the gate
make profile   # per-stage wall clock, NPU vision vs pure CPU
```

Every recipe self-locks the NPU; do not wrap `make` in an outer `flock` on the
same file.
