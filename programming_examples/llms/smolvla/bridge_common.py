# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for the SmolVLA NPU subprocess bridges.

Both `run_npu_backbone.py` and `run_npu_vision.py` are spawned once per
inference by the lerobot-venv driver (the two Python envs are disjoint, so a
single process is impossible). Everything a real single-process deployment
would pay ONCE — safetensors weight load, ELF (re)compile, XRT context + BO
setup — the bridge pays on EVERY call. `ensure_kernels` removes the worst of
it (the multi-minute aircc rebuild) by reusing a complete on-disk ELF cache,
and `Timings` records the remaining phases so the driver can separate
bridge overhead from real NPU compute in the end-to-end report.
"""

import os
import time

_BLAS_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_blas_threads(n=None):
    """Cap the numpy/BLAS thread pool. MUST be called BEFORE numpy is imported.

    Measured on the SmolVLA vision bridge (A3-5 Step 5): the NPU driver loop is
    HOST-BOUND (37 dispatches/image, ~1.75 ms of python+XRT per dispatch), so
    OpenBLAS's worker threads — which busy-spin for a while after finishing the
    one host matmul we do (im2col) — preempt the dispatch thread and stretch the
    encoder from 135 ms to 178 ms per image. Pinning BLAS to 1 thread costs
    ~4 ms on that matmul and buys back ~40 ms per image (~120 ms per inference,
    3 cameras).

    Only the NPU bridge processes do this; the lerobot driver keeps all its
    threads for the CPU action expert, which IS BLAS-bound.
    Override with SMOLVLA_BLAS_THREADS (e.g. 0 = leave the default alone).
    """
    n = n if n is not None else os.environ.get("SMOLVLA_BLAS_THREADS", "1")
    if str(n) == "0":
        return
    for var in _BLAS_VARS:
        os.environ.setdefault(var, str(n))


def ensure_kernels(cache, expected, compile_fn, tag="npu"):
    """Load the cached ELFs if the on-disk cache is COMPLETE, else compile.

    A bridge process is spawned per inference; recompiling every ELF each time
    would dominate any end-to-end measurement and is not something a deployment
    would do (ELFs are build artifacts). We reuse the cache only when its
    manifest resolves AND contains every kernel name in `expected`, so a cache
    from an older/partial kernel set still triggers a full rebuild.

    Set SMOLVLA_FORCE_COMPILE=1 to always rebuild (use after editing any
    kernel builder — the manifest does not track source hashes).

    Returns True if a compile was performed.
    """
    force = os.environ.get("SMOLVLA_FORCE_COMPILE", "0") == "1"
    expected = set(expected)
    if not force and cache.load_manifest() and expected <= set(cache.artifacts):
        print(
            f"[{tag}] reusing {len(cache.artifacts)} cached ELFs from "
            f"{cache.cache_dir} (SMOLVLA_FORCE_COMPILE=1 to rebuild)",
            flush=True,
        )
        return False
    cache.artifacts.clear()
    compile_fn()
    return True


class Timings:
    """Tiny phase timer; `.mark(name)` closes the phase opened by the previous
    mark. Values are milliseconds and are written into the bridge's out-npz as
    `t_<name>_ms` scalars so the driver can attribute wall time."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self._last = self.t0
        self.phases = {}

    def mark(self, name):
        now = time.perf_counter()
        self.phases[name] = (now - self._last) * 1e3
        self._last = now
        return self.phases[name]

    def total_ms(self):
        return (time.perf_counter() - self.t0) * 1e3

    def as_npz_fields(self):
        d = {f"t_{k}_ms": float(v) for k, v in self.phases.items()}
        d["t_bridge_total_ms"] = float(self.total_ms())
        return d

    def report(self, tag):
        parts = ", ".join(f"{k} {v:.1f}ms" for k, v in self.phases.items())
        print(f"[{tag}] phases: {parts} | total {self.total_ms():.1f}ms", flush=True)
