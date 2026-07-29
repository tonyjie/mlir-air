# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-process NPU runtime for SmolVLA's vision encoder.

Why single-process
------------------
An earlier design ran the NPU stage in a *second* Python process, on the
assumption that the lerobot venv could not import `air`/`pyxrt`. It can, once
the mlir-air environment is sourced (`utils/env_setup.sh`), so everything fits
in one process. That matters: the two-process design paid, on EVERY inference,
process spawn + interpreter import (~185 ms) + a safetensors weight reload
(~320-390 ms) + ELF and XRT context load (~34 ms) + an npz round-trip. None of
it is NPU cost. Here the weights and the `KernelCache` (ELFs, XRT context,
device buffer objects) are built ONCE per process and reused, which is what a
deployment does.

Contents
--------
`VisionRuntime`       SigLIP ViT (12 layers, seq 1024) + connector.
                      `encode(images)` -> (N, 64, 960), the RAW connector output
                      that `vlm_with_expert.embed_image` returns. lerobot applies
                      the sqrt(960) scale afterwards -- do not pre-apply it.
`get_vision_runtime()` module-level singleton, so repeated inferences in one
                      process share the weights and the compiled ELFs.
`npu_thread_limits()`  see below.

BLAS-thread contention
----------------------
The NPU driver loop is host-bound (38 dispatches per image). OpenBLAS worker
threads busy-spin after any host matmul in the same process and preempt the
dispatch thread -- measured 135 -> 178 ms per image on the vision encoder.
Clamping the pools globally would be wrong, because the CPU stages that run
before and after this one genuinely want all cores. So the clamp is *scoped*:
`npu_thread_limits()` reduces the pools for the duration of the NPU call and
restores them afterwards, at runtime and without environment variables, by
calling `openblas_set_num_threads` / `mkl_set_num_threads` through ctypes on the
already-loaded shared objects, plus `torch.set_num_threads`.
Set SMOLVLA_NPU_BLAS_LIMIT=0 to disable it and measure the other side.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from shared.infra.cache import KernelCache, Profiler  # noqa: E402

MODEL_ID = "lerobot/smolvla_base"

# ---------------------------------------------------------------------------
# Scoped thread-pool limiting
# ---------------------------------------------------------------------------

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

_blas_controllers = None  # lazily discovered [(handle, set_fn, get_fn_or_None)]


def _discover_blas_controllers():
    """Find the BLAS shared objects already mapped into this process and grab
    their thread-count setter/getter. Mirrors what threadpoolctl does, without
    the dependency (threadpoolctl is not installed in the lerobot venv)."""
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
        set_fn = next(
            (getattr(h, s) for s in _SET_SYMBOLS if hasattr(h, s)),
            None,
        )
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
def npu_thread_limits(n=1, enabled=None):
    """Clamp BLAS + torch intra-op threads to `n` for the duration of the NPU
    dispatch loop, then restore. Scoped, so the CPU action expert outside this
    block keeps every core.

    enabled=None -> honour SMOLVLA_NPU_BLAS_LIMIT (default "1"; "0" disables,
    which is the A/B arm reported in docs/TODO.md)."""
    if enabled is None:
        enabled = os.environ.get("SMOLVLA_NPU_BLAS_LIMIT", "1") != "0"
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


def blas_limiter_available():
    """True if we actually found a BLAS thread knob (else the limiter only
    affects torch). Reported by the bench so the trade-off is auditable."""
    return len(_discover_blas_controllers()) > 0


# ---------------------------------------------------------------------------
# Vision (SigLIP ViT + connector)
# ---------------------------------------------------------------------------

VISION_CACHE_DIR = "vision_kernel_cache"
VISION_SEQ_LEN = 1024
VISION_KERNELS = {
    "vit_ln_qkv",
    "vit_o_ffn",
    "flash_attn",
    "layer_norm",
    "gemm_connector",
}


def ensure_kernels(cache, expected, compile_fn, tag="npu"):
    """Load the cached ELFs if the on-disk cache is COMPLETE, else compile.

    ELFs are build artifacts; recompiling them on every process start would
    dominate any end-to-end measurement and is not what a deployment does. The
    cache is reused only when its manifest resolves AND contains every kernel
    name in `expected`, so a cache left over from an older or partial kernel set
    still triggers a full rebuild rather than silently linking stale objects.

    Set SMOLVLA_FORCE_COMPILE=1 to always rebuild -- necessary after editing any
    kernel builder, since the manifest does not track source hashes.

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


def _to_chw_f32(img):
    """Accept a torch tensor (1,3,H,W) / (3,H,W) or a numpy array; return
    (3,H,W) float32 without importing torch here."""
    if hasattr(img, "detach"):
        arr = img.detach().float().cpu().numpy()
    else:
        arr = np.asarray(img)
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 4:
        assert arr.shape[0] == 1, arr.shape
        arr = arr[0]
    assert arr.ndim == 3 and arr.shape[0] == 3, arr.shape
    return np.ascontiguousarray(arr)


class VisionRuntime:
    """12-layer SigLIP ViT + modality connector on NPU, loaded ONCE.

    The expensive setup — safetensors weight load, ELF cache load, XRT context,
    per-layer static weight BOs — happens in `__init__` / the first `encode`
    and is then reused for every later inference in this process."""

    def __init__(self, cache_dir=VISION_CACHE_DIR, model_id=MODEL_ID, verbose=False):
        from smolvla_vision_weights import load_vision_weights, SigLIPVisionConfig
        from smolvla_vision_npu import compile_all_kernels

        t0 = time.perf_counter()
        self.cfg = SigLIPVisionConfig()
        self.weights = load_vision_weights(model_id, dtype=bfloat16, config=self.cfg)
        t_w = time.perf_counter()
        self.cache = KernelCache(
            str(_HERE / cache_dir), verbose=verbose, profiler=Profiler()
        )
        self.compiled = ensure_kernels(
            self.cache,
            VISION_KERNELS,
            lambda: compile_all_kernels(
                self.cache, self.cfg, VISION_SEQ_LEN, with_connector=True
            ),
            tag="npu-vision",
        )
        t_k = time.perf_counter()
        self.setup_ms = {
            "weight_load": (t_w - t0) * 1e3,
            "kernels": (t_k - t_w) * 1e3,
        }
        self.warmed = False

    def encode(self, images, attn_mode="flash", timings=None, limit_threads=None):
        """images: sequence of N camera tensors/arrays, each (1,3,512,512) or
        (3,512,512), ALREADY lerobot-preprocessed (resize_with_pad + [-1,1]).
        Returns (N, 64, 960) f32 RAW connector output."""
        from smolvla_vision_npu import run_vit_encoder

        from smolvla_cpu_helpers import im2col_patch_embed

        t0 = time.perf_counter()
        arrs = [_to_chw_f32(im) for im in images]
        # The im2col patch-embed is a ONE-TIME host matmul per image, not part
        # of the dispatch loop, so it runs BEFORE the clamp with all threads
        # (measured 4.6 ms multi-threaded vs 9.6 ms at 1 thread, per image).
        patch_embeds = [
            im2col_patch_embed(
                a,
                self.weights.patch_w,
                self.weights.patch_b,
                self.weights.pos_embed,
                self.cfg.patch_size,
            )
            for a in arrs
        ]
        t_im2col = (time.perf_counter() - t0) * 1e3
        out = np.empty((len(arrs), 64, self.cfg.connector_out), np.float32)
        per_image_ms = []
        with npu_thread_limits(1, enabled=limit_threads):
            t_enc0 = time.perf_counter()
            for i, a in enumerate(patch_embeds):
                ti = time.perf_counter()
                res = run_vit_encoder(
                    a,
                    self.weights,
                    self.cfg,
                    self.cache,
                    return_per_layer=False,
                    do_connector=True,
                    verbose=False,
                    attn_mode=attn_mode,
                )
                out[i] = res["connector"]
                per_image_ms.append((time.perf_counter() - ti) * 1e3)
            t_enc = (time.perf_counter() - t_enc0) * 1e3
        self.warmed = True
        if timings is not None:
            timings["vision"] = {
                "wall_ms": (time.perf_counter() - t0) * 1e3,
                "t_im2col_ms": t_im2col,
                "t_encode_ms": t_enc,
                "t_image_ms": per_image_ms,
            }
        return out

    def warmup(self):
        """One throwaway encode so the first *measured* inference does not pay
        XRT context creation + BO alloc + the one-time static weight upload
        (measured 335 ms vs 148 ms warm per image)."""
        if not self.warmed:
            self.encode([np.zeros((3, 512, 512), np.float32)])
        return self


_vision_rt = None


def get_vision_runtime(**kw):
    """The whole point of single-process: build the vision runtime once and
    reuse its weights + warm KernelCache for every later inference."""
    global _vision_rt
    if _vision_rt is None:
        _vision_rt = VisionRuntime(**kw)
    return _vision_rt
