# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoped host thread-pool clamping for in-process NPU dispatch loops.

The problem this solves is not specific to any model. When the NPU stage runs
in the SAME process as host-side PyTorch/numpy work, the driver dispatch loop
is host-bound, and OpenBLAS worker threads busy-spin after any host matmul and
preempt the dispatch thread. Measured on SmolVLA's vision encoder (38
dispatches per image): 135 -> 178 ms per image, a 32% loss to threads that are
doing nothing.

Clamping the pools globally is the wrong fix, because the CPU stages before and
after the NPU call genuinely want every core. So the clamp is SCOPED:
`host_thread_limits()` reduces the pools for the duration of a `with` block and
restores them afterwards -- at runtime, through ctypes on the already-loaded
shared objects, so it works after numpy/torch have been imported (environment
variables like OMP_NUM_THREADS only work before).

This mirrors what `threadpoolctl` does, without the dependency, which matters
when the model's venv does not have it installed.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from contextlib import contextmanager

# Symbol spellings seen in the wild. numpy>=2 wheels bundle scipy-openblas64,
# whose setter is `scipy_openblas_set_num_threads64_`; conda/pip OpenBLAS uses
# the plain name; MKL builds export MKL_Set_Num_Threads.
_SET_SYMBOLS = (
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads_64_",
    "scipy_openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
    "MKL_Set_Num_Threads",
    "mkl_set_num_threads",
)
_GET_SYMBOLS = (
    "scipy_openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads_64_",
    "scipy_openblas_get_num_threads",
    "openblas_get_num_threads64_",
    "openblas_get_num_threads",
    "MKL_Get_Max_Threads",
    "mkl_get_max_threads",
)
_LIB_PATTERN = re.compile(r"(openblas|mkl_rt|libmkl_core)", re.IGNORECASE)

_blas_controllers = None  # lazily discovered [(path, set_fn, get_fn_or_None)]


def _discover_blas_controllers():
    """Find the BLAS shared objects already mapped into this process and grab
    their thread-count setter/getter."""
    global _blas_controllers
    if _blas_controllers is not None:
        return _blas_controllers
    ctrls = []
    seen = set()
    try:
        with open("/proc/self/maps") as f:
            lines = f.readlines()
    except OSError:
        lines = []
    for line in lines:
        m = re.search(r"(/\S+\.so[^\s]*)", line)
        if not m:
            continue
        path = m.group(1)
        if path in seen or not _LIB_PATTERN.search(path):
            continue
        seen.add(path)
        try:
            # RTLD_NOLOAD: bind to the ALREADY-mapped library (the one numpy is
            # actually using); never dlopen a second copy.
            h = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOLOAD", 0))
        except OSError:
            continue
        set_fn = next((getattr(h, s) for s in _SET_SYMBOLS if hasattr(h, s)), None)
        if set_fn is None:
            continue
        get_fn = next((getattr(h, s) for s in _GET_SYMBOLS if hasattr(h, s)), None)
        ctrls.append((path, set_fn, get_fn))
    _blas_controllers = ctrls
    return ctrls


def _set_blas_threads(n):
    """-> list of (set_fn, previous_n) so the caller can restore."""
    prev = []
    for _path, set_fn, get_fn in _discover_blas_controllers():
        try:
            old = int(get_fn()) if get_fn is not None else (os.cpu_count() or 1)
            set_fn(ctypes.c_int(int(n)))
            prev.append((set_fn, old))
        except Exception:  # never let a thread knob break inference
            continue
    return prev


@contextmanager
def host_thread_limits(n=1, enabled=True):
    """Clamp BLAS + torch intra-op threads to `n` inside the block, restore after.

    Scoped on purpose: the CPU stages outside the block keep every core. Torch
    is only touched if it is already imported -- this never imports it.
    """
    if not enabled:
        yield
        return
    prev = _set_blas_threads(n)
    torch_prev = None
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            torch_prev = torch_mod.get_num_threads()
            torch_mod.set_num_threads(int(n))
        except Exception:
            torch_prev = None
    try:
        yield
    finally:
        for set_fn, old in prev:
            try:
                set_fn(ctypes.c_int(int(old)))
            except Exception:
                pass
        if torch_prev is not None:
            try:
                torch_mod.set_num_threads(torch_prev)
            except Exception:
                pass
