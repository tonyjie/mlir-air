# SmolVLA on NPU2 — usage guide

## Prerequisites

### Hardware and toolchain
- AMD NPU2 hardware (Strix, AIE2P)
- MLIR-AIR installed with the Peano compiler (`PEANO_INSTALL_DIR` set)
- The project's standard environment: `source utils/env_setup.sh ...`

### Python environment
This example needs **one** interpreter that has both sides:

- `torch` + `lerobot` — to run the real SmolVLA policy
- `air` + `pyxrt` — to drive the NPU

That combination is what makes the single-process path work. `lerobot` is in
`requirements.txt` — it IS the CPU baseline this example verifies against, so
nothing needs to be fetched by hand.

Install into the environment that already has `air` + `pyxrt`:

```bash
pip install -r requirements.txt
make verify                      # default LEROBOT_PYTHON=python3
```

**One caveat.** lerobot requires `numpy>=2.0,<2.3`, and `utils/requirements.txt`
leaves numpy unpinned, so an older mlir-air environment may still be on 1.x —
installing will upgrade it across a major version in the interpreter every
other example shares. That upgrade was tested here (1.26.4 → 2.2.6, with
`qwen25_0_5b` and `qwen3_1_7b` `make verify` PASSing identically before and
after), but check your own models rather than taking that on faith.

To leave the shared environment untouched, use a separate venv instead.
Sourcing the mlir-air environment puts `air` and `pyxrt` on `PYTHONPATH` /
`LD_LIBRARY_PATH`, so they import from any venv:

```bash
python3 -m venv ~/smolvla-venv
~/smolvla-venv/bin/pip install -r requirements.txt
make verify LEROBOT_PYTHON=~/smolvla-venv/bin/python
```

### Model access
`HF_TOKEN` must be set; the example downloads `lerobot/smolvla_base` (450M
parameters) on first use.

---

## Targets

```bash
make help      # this list
make compile   # build every vision ELF — no NPU dispatch, no download
make cpu-baseline  # run the unmodified CPU model on its own, for inspection (no NPU)
make run       # one end-to-end forward; prints the action chunk
make verify    # THE GATE — action chunk vs the pure-CPU model (PASS/FAIL)
make profile   # CPU vs NPU, interleaved and warmed, with the per-ELF breakdown
make clean     # remove the kernel cache and build artifacts
```

### The NPU lock
On a machine where several sessions share one NPU, take the project's lock
around anything that touches the device — the same convention the sibling
examples use:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

The recipes deliberately do **not** take that lock themselves; if they did,
the command above would deadlock against itself.

Correctness does not depend on it: `shared/infra/cache.py` holds
`/tmp/npu.lock` around every dispatch, so concurrent runs interleave safely.
The outer lock matters for *timing* — it stops someone else's dispatches from
landing in the middle of your 38.

---

## First run

```bash
make compile        # a few minutes; produces vision_kernel_cache/
make verify         # no fixture needed — the gate computes its CPU reference live
```

Expected:

```
==================================================================
SmolVLA verify: end-to-end action-chunk regression gate
  NPU stages     : vision
  execution model: single-process (air/pyxrt in the lerobot venv)
==================================================================
  cosine   = 0.998996
  cos_min  = 0.99
  nmse     = 0.003023
  nmse_max = 0.04
  passed   = True
==================================================================
[verify] PASS
```

`make verify --cpu-vision` runs the unmodified model against its own baseline
and should score exactly 1.0 — a sanity check of the harness itself.

---

## Rebuilding kernels after an edit

The ELF cache is reused whenever its manifest resolves and contains every
expected kernel. The manifest does **not** track source hashes, so after editing
any kernel builder you must force a rebuild:

```bash
SMOLVLA_FORCE_COMPILE=1 make verify
```

---

## Environment variables

| Variable | Effect |
|---|---|
| `LEROBOT_PYTHON` | interpreter with torch + lerobot + air + pyxrt |
| `HF_TOKEN` | required for the checkpoint download |
| `SMOLVLA_FORCE_COMPILE=1` | rebuild every ELF instead of reusing the cache |
| `SMOLVLA_NPU_BLAS_LIMIT=0` | disable the scoped BLAS-thread clamp around the NPU call (costs ~25–70 ms per inference; see `explain.md` §6) |

---

## Measuring honestly

Two things will corrupt a timing run on a shared machine:

1. **Other processes on the NPU.** `flock` is advisory — it only protects
   against processes that also take the same lock. Check with
   `fuser /dev/accel/accel0`.
2. **CPU/NPU power state.** Every number in this example was measured with the
   CPU governor and EPP at `performance` and the NPU at `pmode=Turbo`. A
   `balanced` machine reads very differently, and the CPU baseline moves more
   than the NPU stage does — which changes the *ratio*, not just the absolutes.

`make profile` runs the CPU and NPU configurations back to back under one lock
so the two are comparable.
