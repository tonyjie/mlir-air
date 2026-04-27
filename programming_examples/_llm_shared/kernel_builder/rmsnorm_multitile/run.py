#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm multi-tile (herd_x=8) — standalone correctness + profile harness.

The existing `weighted_rms_norm/weighted_rms_norm.py:build_module(herd_x>1)`
emits a **bare `air.herd`** (no air.launch / air.segment wrapper) intended
for stitching into multi-launch ELFs (e.g. llama3's rms_gemms_rope_multi).
That bare herd doesn't compile to a valid standalone NPU kernel — it produces
all-zero output (the lowering's `airrt-to-npu` pass drops it because there's
no `airrt.segment_load`).

This harness wraps the bare herd in launch+segment via `_wrap_ir_in_launch`
(reused from `_llm_shared/kernel_builder/stitching.py`) so the multi-tile
path can be tested standalone at llama3 production shape.

Default shape mirrors llama3 final-norm (and per-block norms):
    M = 2048   (seq_len)
    N = 2048   (emb_dim)
    herd_x = 8

Tolerances reported with each run:
    rtol = 5e-2 (5%, matches the existing weighted_rms_norm.py main runner)
    atol = 5e-1 (matches existing)
    min_correlation = 0.99
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

# sys.path: programming_examples/ + _llm_shared/kernel_builder/ +
# weighted_rms_norm/ (for build_module + reference).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))  # programming_examples/
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # _llm_shared/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "weighted_rms_norm"))

from air.ir import Module, Context

from weighted_rms_norm import build_module, rms_norm_reference
from _llm_shared.kernel_builder.stitching import _wrap_ir_in_launch

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend


def build_wrapped_module(M, N, vector_size=16, herd_x=8):
    """Build RMSNorm at multi-tile config and wrap bare herd in launch+segment.

    Returns a parsed air.ir.Module ready for compilation.
    """
    # Original module — has bare herd because herd_x > 1
    bare_module = build_module(M, N, bfloat16, vector_size, herd_x=herd_x)
    bare_text = str(bare_module)

    # Wrap with launch + segment
    wrapped_text = _wrap_ir_in_launch(bare_text)

    # Re-parse into a fresh Context
    with Context() as ctx:
        return Module.parse(wrapped_text, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RMSNorm multi-tile (herd_x>=2) standalone correctness + profile"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Run profile mode (5 warmup + 20 iter)"
    )
    parser.add_argument(
        "--M", type=int, default=2048, help="Rows (default: LLAMA seq_len)"
    )
    parser.add_argument(
        "--N", type=int, default=2048, help="Cols (default: LLAMA emb_dim)"
    )
    parser.add_argument(
        "--herd-x",
        type=int,
        default=8,
        help="Herd width (default: 8 = llama3 production)",
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument(
        "--iterations", type=int, default=20, help="Profile timed iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Profile warmup iterations"
    )
    args = parser.parse_args()

    M, N, herd_x = args.M, args.N, args.herd_x
    print(
        f"Weighted RMSNorm (multi-tile via launch wrapper): M={M}, N={N}, herd=[{herd_x}, 1]"
    )

    module = build_wrapped_module(M, N, args.vector_size, herd_x=herd_x)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data — match weighted_rms_norm.py's main runner conventions
    np.random.seed(0)
    x_input = np.random.rand(M, N).astype(bfloat16)
    weight = np.random.rand(N).astype(bfloat16)
    y_expected = rms_norm_reference(x_input, weight)

    # Tolerances (documented + enforced)
    RTOL = 5e-2  # 5% relative — same as weighted_rms_norm.py main runner
    ATOL = 5e-1  # 0.5 absolute — same
    MIN_CORR = 0.99
    print(f"  Tolerances: rtol={RTOL}, atol={ATOL}, min_correlation={MIN_CORR}")

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="weighted_rms_norm",
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        out_buf = np.zeros((M, N), dtype=bfloat16)
        inputs = [x_input, weight, out_buf]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        # Warmup (untimed)
        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for _ in range(args.warmup):
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()

        # Timed iterations
        times_kernel, times_total = [], []
        for it in range(args.iterations):
            t0 = time.perf_counter()
            for i, a in enumerate(inputs):
                bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            tk0 = time.perf_counter()
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()
            tk1 = time.perf_counter()
            for bo in bos:
                bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            t1 = time.perf_counter()
            times_kernel.append((tk1 - tk0) * 1000)
            times_total.append((t1 - t0) * 1000)

        # Read output + check correctness
        output_data = bos[-1].read(sizes[-1], 0).view(np.int16).view(bfloat16)
        output_data = output_data.reshape(M, N).astype(np.float32)
        ref_flat = y_expected.astype(np.float32).flatten()
        out_flat = output_data.flatten()

        cosine = float(
            np.dot(out_flat, ref_flat)
            / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-12)
        )
        max_abs = float(np.abs(out_flat - ref_flat).max())
        max_rel = float((np.abs(out_flat - ref_flat) / (np.abs(ref_flat) + 1e-6)).max())

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.warmup} warmup + {args.iterations} iter)")
        print(f"{'='*60}")
        print(
            f"  Kernel:           avg={np.mean(times_kernel):.3f}ms  "
            f"min={np.min(times_kernel):.3f}ms  max={np.max(times_kernel):.3f}ms"
        )
        print(
            f"  Total (w+r+rd):   avg={np.mean(times_total):.3f}ms  "
            f"min={np.min(times_total):.3f}ms  max={np.max(times_total):.3f}ms"
        )
        print(
            f"  Host overhead:    {np.mean(times_total) - np.mean(times_kernel):.3f}ms"
        )
        print(f"\n  Cosine:           {cosine:.6f} (threshold: {MIN_CORR})")
        print(f"  Max abs error:    {max_abs:.4f} (threshold atol: {ATOL})")
        print(f"  Max rel error:    {max_rel:.4f} (threshold rtol: {RTOL})")
        status = "PASS" if cosine >= MIN_CORR else "FAIL"
        print(f"\n  {status} (cosine={cosine:.6f})")

    else:
        # Correctness mode — XRTRunner does the rtol/atol check + correlation
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="weighted_rms_norm",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                module,
                inputs=[x_input, weight],
                expected_outputs=[y_expected],
                rtol=RTOL,
                atol=ATOL,
                min_correlation=MIN_CORR,
            )
        )
