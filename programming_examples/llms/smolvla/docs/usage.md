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

That combination is what makes the single-process path work. A lerobot venv
qualifies once the mlir-air environment is sourced, because the mlir-air
`PYTHONPATH` / `LD_LIBRARY_PATH` make `air` and `pyxrt` importable from it.

```bash
pip install -r requirements.txt
```

Point the Makefile at it if it is not at the default location:

```bash
make verify LEROBOT_PYTHON=/path/to/venv/bin/python
```

### Model access
`HF_TOKEN` must be set; the example downloads `lerobot/smolvla_base` (450M
parameters) on first use.

---

## Targets

```bash
make help      # this list
make compile   # build every vision ELF — no NPU dispatch, no download
make oracle    # regenerate smolvla_oracle.npz, the pure-CPU baseline (no NPU)
make run       # one end-to-end forward; prints the action chunk
make verify    # THE GATE — action chunk vs the pure-CPU model (PASS/FAIL)
make profile   # per-stage wall clock, NPU vision vs pure CPU
make clean     # remove the kernel cache and build artifacts
```

### The NPU lock
Every recipe that touches the device already wraps itself in
`flock /tmp/mlir-air-npu.lock`. **Do not** wrap `make` in an outer `flock` on
the same file — it self-deadlocks. If you invoke the Python entry points
directly, add the lock yourself:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 smolvla_inference.py
```

---

## First run

```bash
make compile        # a few minutes; produces vision_kernel_cache/
make oracle         # CPU only; produces smolvla_oracle.npz
make verify
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
