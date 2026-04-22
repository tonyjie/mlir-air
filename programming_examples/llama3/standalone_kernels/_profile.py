# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared profiling helper for chunked-prefill standalone kernel tests.

Each kernel test imports `profile_kernel` and passes:
  - build_fn: zero-arg callable returning the MLIR module
  - inputs: list of numpy arrays (host-side data passed to the kernel as-is)
  - gflops_per_invocation: theoretical FLOP count for one kernel invocation
  - herd_active, herd_total: for utilization reporting (e.g. (1*4, 8*4))
  - instance_name: distinguishes parallel build dirs

Reports avg/min/max latency, achieved GFLOPS (using min latency), and core
utilization headroom relative to the full 8x4 herd.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
from ml_dtypes import bfloat16


def profile_kernel(
    build_fn: Callable[[], object],
    *,
    inputs: Iterable[np.ndarray],
    gflops_per_invocation: float,
    herd_active: int,
    herd_total: int = 32,
    instance_name: str,
    label: str,
    iterations: int = 20,
    warmup: int = 5,
    expected_output: np.ndarray | None = None,
    output_index: int = -1,
    correlation_threshold: float = 0.99,
) -> int:
    """Compile via XRTBackend, run kernel `iterations` times, report perf
    and (if `expected_output` is given) correctness vs CPU reference.

    Args:
        inputs: list of numpy arrays passed to the kernel as args. The slot
            indexed by `output_index` is also where the kernel's output gets
            written; we read it back for correlation after timing.
        expected_output: CPU reference for correctness verification. If None,
            no correlation check is performed.
        output_index: index of the output buffer in `inputs` (default: last).
        correlation_threshold: pearson correlation below this prints WARN.

    Returns:
        0 on success (timing OK + correlation >= threshold if check enabled),
        non-zero if correlation check failed.
    """
    import filelock
    import pyxrt as xrt

    from air.backend.xrt import XRTBackend

    inputs = list(inputs)
    print(
        f"{label} profile: inputs={[a.shape for a in inputs]}  "
        f"iters={iterations} (warmup={warmup})"
    )

    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name=instance_name,
    )
    artifact = backend.compile(build_fn())
    out_idx = output_index if output_index >= 0 else len(inputs) + output_index
    with filelock.FileLock("/tmp/npu.lock"):
        backend.load(artifact)

        sizes = [arr.size * arr.itemsize for arr in inputs]
        bos = [
            xrt.bo(
                backend.device,
                s,
                xrt.bo.host_only,
                backend.kernel.group_id(i + 3),
            )
            for i, s in enumerate(sizes)
        ]
        for i, arr in enumerate(inputs):
            view = arr.view(np.int16) if arr.dtype == bfloat16 else arr
            bos[i].write(view, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        backend.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        for _ in range(warmup):
            h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
            h.wait()

        times_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            h = backend.kernel(3, backend.bo_instr, len(backend.instr_v), *bos)
            h.wait()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        # Read back the output BO for correlation (after timing loop).
        bos[out_idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out_bytes = bos[out_idx].read(sizes[out_idx], 0)

    backend.unload()

    avg_ms = float(np.mean(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    achieved_gflops = gflops_per_invocation / (min_ms / 1000.0)
    util_pct = 100.0 * herd_active / herd_total

    print(f"  Compute (per invocation): {gflops_per_invocation:.3f} GFLOP")
    print(
        f"  Latency:                  avg={avg_ms:.2f} ms  min={min_ms:.2f} ms  max={max_ms:.2f} ms"
    )
    print(f"  Achieved (using min):     {achieved_gflops:.1f} GFLOPS")
    print(f"  Cores active:             {herd_active}/{herd_total} ({util_pct:.1f}%)")
    if herd_active < herd_total:
        print(
            f"  Headroom if full {herd_total}-core herd: "
            f"~{herd_total / herd_active:.1f}x potential speedup"
        )

    # Correlation check (if expected_output given).
    if expected_output is None:
        return 0

    out_arr = inputs[out_idx]
    if out_arr.dtype == bfloat16:
        actual = (
            np.frombuffer(out_bytes, dtype=np.int16)
            .view(bfloat16)
            .reshape(out_arr.shape)
        )
    else:
        actual = np.frombuffer(out_bytes, dtype=out_arr.dtype).reshape(out_arr.shape)
    actual_f32 = actual.astype(np.float32).flatten()
    expected_f32 = expected_output.astype(np.float32).flatten()
    corr = (
        float(np.corrcoef(actual_f32, expected_f32)[0, 1])
        if expected_f32.size > 1
        else float("nan")
    )
    max_err = float(np.max(np.abs(actual_f32 - expected_f32)))
    mean_err = float(np.mean(np.abs(actual_f32 - expected_f32)))
    status = "OK" if corr >= correlation_threshold else "WARN"
    print(
        f"  Correlation:              [{status}] corr={corr:.6f}  "
        f"max_abs_err={max_err:.4f}  mean_abs_err={mean_err:.4f}"
    )
    return 0 if corr >= correlation_threshold else 1
