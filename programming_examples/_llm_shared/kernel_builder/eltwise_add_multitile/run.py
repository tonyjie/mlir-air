#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eltwise Add (residual) — standalone correctness + profile harness.

Llama3-1B uses `eltwise_add.build_module` 4 times per transformer layer:
  - Prefill: post-attn residual + FFN-out residual (both via custom 2D
    wrappers in `o_ffn_multi.py`)
  - Decode:  same two residuals (via direct `_wrap_ir_in_launch` on the
    bare-herd output of `eltwise_add.build_module(n=2048, tile_n=256,
    herd_x=8, herd_y=1)`, in `o_gemv_ffn_multi.py`)

This harness exposes the decode-shape (n=2048, herd_x=8) standalone via
the same `_wrap_ir_in_launch` recipe.

Tolerances:
    rtol = 1e-3, atol = 1e-3, min_correlation = 0.9999
    (eltwise add has no accumulation, so BF16 round-trip noise should be
    well under 0.1%.)
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))  # programming_examples/
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # _llm_shared/

from air.ir import Module, Context

from eltwise_add.eltwise_add import build_module as build_add
from _llm_shared.kernel_builder.stitching import _wrap_ir_in_launch

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend


def build_wrapped_module(n=2048, tile_n=256, herd_x=8, herd_y=1):
    """Build eltwise_add at the multi-tile config and wrap bare herd in launch+segment."""
    bare = build_add(n, tile_n, bfloat16, vector_size=16, herd_x=herd_x, herd_y=herd_y)
    wrapped = _wrap_ir_in_launch(str(bare))
    with Context() as ctx:
        return Module.parse(wrapped, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eltwise Add multi-tile (herd_x>=2) correctness + profile"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--n", type=int, default=2048, help="Total elements (default: LLAMA emb_dim)"
    )
    parser.add_argument(
        "--tile-n", type=int, default=None, help="Tile size (default: n // 8)"
    )
    parser.add_argument("--herd-x", type=int, default=8)
    parser.add_argument("--herd-y", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    n = args.n
    tile_n = args.tile_n if args.tile_n is not None else n // args.herd_x
    print(f"Eltwise Add: n={n}, tile_n={tile_n}, herd=[{args.herd_x},{args.herd_y}]")

    module = build_wrapped_module(n, tile_n, args.herd_x, args.herd_y)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(0)
    a = np.random.uniform(-1, 1, (n,)).astype(bfloat16)
    b = np.random.uniform(-1, 1, (n,)).astype(bfloat16)
    c_ref = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)

    RTOL, ATOL, MIN_CORR = 1e-2, 1e-2, 0.9999  # BF16: ~4e-3 ulp; 1% rtol is comfortable
    print(f"  Tolerances: rtol={RTOL}, atol={ATOL}, min_correlation={MIN_CORR}")

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="eltwise_add",
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module)
        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        c_buf = np.zeros((n,), dtype=bfloat16)
        inputs = [a, b, c_buf]
        sizes = [x.size * x.itemsize for x in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        for i, x in enumerate(inputs):
            bos[i].write(x.view(np.int16) if x.dtype == bfloat16 else x, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for _ in range(args.warmup):
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()

        times_kernel, times_total = [], []
        for it in range(args.iterations):
            t0 = time.perf_counter()
            for i, x in enumerate(inputs):
                bos[i].write(x.view(np.int16) if x.dtype == bfloat16 else x, 0)
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

        out = bos[-1].read(sizes[-1], 0).view(np.int16).view(bfloat16).reshape(n)
        out_f, ref_f = out.astype(np.float32), c_ref.astype(np.float32)
        cosine = float(
            np.dot(out_f, ref_f)
            / (np.linalg.norm(out_f) * np.linalg.norm(ref_f) + 1e-12)
        )
        max_abs = float(np.abs(out_f - ref_f).max())
        max_rel = float((np.abs(out_f - ref_f) / (np.abs(ref_f) + 1e-6)).max())
        backend.unload()

        print(
            f"\n{'='*60}\nPROFILING ({args.warmup} warmup + {args.iterations} iter)\n{'='*60}"
        )
        print(
            f"  Kernel:    avg={np.mean(times_kernel):.3f}ms  min={np.min(times_kernel):.3f}ms  max={np.max(times_kernel):.3f}ms"
        )
        print(f"  Total:     avg={np.mean(times_total):.3f}ms")
        print(f"  Cosine:    {cosine:.6f}  (threshold {MIN_CORR})")
        print(f"  Max abs:   {max_abs:.4f}  (threshold atol {ATOL})")
        print(f"  Max rel:   {max_rel:.4f}  (threshold rtol {RTOL})")
        status = "PASS" if cosine >= MIN_CORR else "FAIL"
        print(f"  → {status}")
    else:
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="eltwise_add",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                module,
                inputs=[a, b],
                expected_outputs=[c_ref],
                rtol=RTOL,
                atol=ATOL,
                min_correlation=MIN_CORR,
            )
        )
