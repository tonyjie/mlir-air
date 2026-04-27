#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eltwise Add prefill 2D variants — standalone correctness + profile harness.

Llama3-1B prefill `o_ffn.elf` uses two custom 2D Eltwise Add builders
(NOT `eltwise_add.build_module`):

  - L2 post-attn residual: `_build_add_2d_to_2d` — 3 args all 2D
    `(rows, cols)`; collapses inside launch for DMA, output stays 2D so
    the next launch can read it as 2D directly.
  - L8 FFN residual:       `_build_add_2d_to_1d` — 2 inputs 2D, output 1D.

Both define complete `air.launch + air.segment + air.herd` (NOT bare
herds), so they are XRTRunner-compatible directly.

Default shape mirrors llama3-1B prefill (rows=2048, cols=2048).

Tolerances (BF16 add, ~4e-3 ulp):
    rtol = 1e-2, atol = 1e-2, min_correlation = 0.9999
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
sys.path.insert(
    0, os.path.join(_HERE, "..", "..", "..", "llama3")
)  # llama3 multi_launch_builder imports

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    subview,
    collapse_shape as memref_collapse_shape,
)
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

# Top-level builder — directly importable from llama3
from llama3.multi_launch_builder.o_ffn_multi import _build_add_2d_to_2d

# ---------------------------------------------------------------------------
# Build module — wrapper that picks 2d_to_2d (imported) or 2d_to_1d (copied)
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_1d(seq_len, emb_dim, np_dtype=bfloat16, vector_size=16):
    """Copy of `o_ffn_multi.py:_build_add_2d_to_1d` (was nested closure;
    parameterized here for standalone use). 2 args 2D + 1 arg 1D output."""
    xrt_dtype = type_mapper(np_dtype)
    n_total = seq_len * emb_dim
    l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
    l3_1d_ty = MemRefType.get([n_total], xrt_dtype)
    total_tiles = 8
    chunk_size = n_total // total_tiles
    tile_n = emb_dim
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
    def eltwise_add(a_2d, b_2d, out_1d):
        @launch(operands=[a_2d, b_2d, out_1d])
        def add_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

            @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
            def add_seg(s_a, s_b, s_out):
                offset_map = AffineMap.get(
                    0,
                    3,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineExpr.get_add(
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(1),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineConstantExpr.get(chunk_size),
                            ),
                        )
                    ],
                )

                @herd(name="add_herd", sizes=[8, 1], operands=[s_a, s_b, s_out])
                def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1_ty, [], [])
                    l1_b = AllocOp(l1_ty, [], [])
                    l1_out = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    for loop_iv in for_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in for_(0, tile_n, 16):
                            sub_a = subview(l1_a.result, [j], [16], [1])
                            sub_b = subview(l1_b.result, [j], [16], [1])
                            sub_out = subview(l1_out.result, [j], [16], [1])
                            v_a = transfer_read(
                                vec_ty, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vec_ty, sub_b, [c0], identity_map, cst0, [True]
                            )
                            v_sum = arith.addf(v_a, v_b)
                            transfer_write(
                                None, v_sum, sub_out, [c0], identity_map, [True]
                            )
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eltwise Add 2D variants (prefill) correctness + profile"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", choices=["2d_to_2d", "2d_to_1d"], default="2d_to_2d")
    parser.add_argument(
        "--rows", type=int, default=2048, help="seq_len (default: LLAMA)"
    )
    parser.add_argument(
        "--cols", type=int, default=2048, help="emb_dim (default: LLAMA)"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    rows, cols = args.rows, args.cols
    print(f"Eltwise Add 2D ({args.mode}): rows={rows}, cols={cols}")

    if args.mode == "2d_to_2d":
        module = _build_add_2d_to_2d(
            rows, cols, bfloat16, vector_size=16, herd_x=8, herd_y=1
        )
        # MLIR func name from `_build_add_2d_to_2d` is `eltwise_add_2d`
        kernel_instance_name = "eltwise_add_2d"
    else:
        module = _build_add_2d_to_1d(rows, cols, bfloat16, vector_size=16)
        # MLIR func name from `_build_add_2d_to_1d` is `eltwise_add`
        kernel_instance_name = "eltwise_add"

    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(0)
    a = np.random.uniform(-1, 1, (rows, cols)).astype(bfloat16)
    b = np.random.uniform(-1, 1, (rows, cols)).astype(bfloat16)
    c_ref_2d = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)
    if args.mode == "2d_to_2d":
        c_ref = c_ref_2d
        out_buf = np.zeros((rows, cols), dtype=bfloat16)
    else:
        c_ref = c_ref_2d.reshape(-1)
        out_buf = np.zeros((rows * cols,), dtype=bfloat16)

    RTOL, ATOL, MIN_CORR = 1e-2, 1e-2, 0.9999
    print(f"  Tolerances: rtol={RTOL}, atol={ATOL}, min_correlation={MIN_CORR}")

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name=kernel_instance_name,
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [a, b, out_buf]
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

        out = bos[-1].read(sizes[-1], 0).view(np.int16).view(bfloat16)
        out = (
            out.reshape((rows, cols))
            if args.mode == "2d_to_2d"
            else out.reshape(rows * cols)
        )
        out_f, ref_f = (
            out.astype(np.float32).flatten(),
            c_ref.astype(np.float32).flatten(),
        )
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
            instance_name=kernel_instance_name,
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
