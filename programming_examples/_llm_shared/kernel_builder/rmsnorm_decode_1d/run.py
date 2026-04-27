#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm 1D (decode-shape, M=1) — standalone correctness + profile harness.

Llama3-1B decode uses a different RMSNorm builder than prefill:
`_build_rms_1d(n)` in `llama3/multi_launch_builder/rms_gemv_rope_multi.py`.

The 1D builder takes 1D `memref<N x bf16>` func args (matching the GEMV
input layout downstream) and uses `expand_shape` 1D → (1, N) inside the
launch to drive a single-tile (herd 1×1) M=1 RMSNorm body.

This harness copies `_build_rms_1d` self-contained so it runs without
llama3 dependencies.

Tolerances:
    rtol = 5e-2, atol = 5e-1, min_correlation = 0.99
    (matches the existing weighted_rms_norm.py main runner)
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
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "weighted_rms_norm"))

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    subview,
    expand_shape as memref_expand_shape,
)
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from weighted_rms_norm import rms_norm_reference

EPS = 1e-5


# ---------------------------------------------------------------------------
# Module builder — copied verbatim from llama3/multi_launch_builder/
#                   rms_gemv_rope_multi.py:_build_rms_1d
# ---------------------------------------------------------------------------


@module_builder
def build_module(n, np_dtype=bfloat16, vector_size=16):
    """RMSNorm 1D (M=1) — 1D func args, expand_shape inside launch."""
    xrt_dtype = type_mapper(np_dtype)
    assert n % vector_size == 0

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    l3_2d_ty = MemRefType.get([1, n], xrt_dtype)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([n], xrt_dtype, memory_space=l1_mem_space)
    l1VecTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

    @FuncOp.from_py_func(l3_1d_ty, l3_1d_ty, l3_1d_ty)
    def rms_norm_1d(x_1d, weight, out_1d):
        @launch(operands=[x_1d, weight, out_1d])
        def rms_launch(l_x_1d, l_weight, l_out_1d):
            l_x_2d = memref_expand_shape(l3_2d_ty, l_x_1d, [[0, 1]], [], [1, n])
            l_out_2d = memref_expand_shape(l3_2d_ty, l_out_1d, [[0, 1]], [], [1, n])

            @segment(name="rms_seg", operands=[l_x_2d, l_weight, l_out_2d])
            def rms_seg(s_x_2d, s_weight, s_out_2d):
                @herd(
                    name="rms_herd", sizes=[1, 1], operands=[s_x_2d, s_weight, s_out_2d]
                )
                def rms_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                    l1_row = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])
                    l1_weight_buf = AllocOp(l1RowTy, [], [])
                    l1_acc = AllocOp(l1VecTy, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                    n_f = arith.ConstantOp(xrt_dtype, float(n))
                    eps_f = arith.ConstantOp(xrt_dtype, EPS)
                    v_zero = BroadcastOp(vecTy, cst0)

                    dma_memcpy_nd(l1_weight_buf, l3_weight)

                    row = arith.ConstantOp.create_index(0)

                    dma_memcpy_nd(
                        l1_row,
                        l3_in,
                        src_offsets=[row, 0],
                        src_sizes=[1, n],
                        src_strides=[n, 1],
                    )

                    transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                    for j in for_(0, n, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_tmp = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_sq = arith.mulf(v_x, v_x)
                        transfer_write(None, v_sq, sub_tmp, [c0], identity_map, [True])
                        v_sq_rd = transfer_read(
                            vecTy, sub_tmp, [c0], identity_map, cst0, [True]
                        )
                        v_acc = transfer_read(
                            vecTy, l1_acc, [c0], identity_map, cst0, [True]
                        )
                        v_sum = arith.addf(v_acc, v_sq_rd)
                        transfer_write(None, v_sum, l1_acc, [c0], identity_map, [True])
                        yield_([])

                    v_final = transfer_read(
                        vecTy, l1_acc, [c0], identity_map, cst0, [True]
                    )
                    total_sum = vector_reduction(xrt_dtype, "add", v_final)
                    rms = arith.divf(total_sum, n_f)

                    f32 = F32Type.get()
                    rms_eps = arith.addf(rms, eps_f)
                    rms_eps_f32 = arith.extf(f32, rms_eps)
                    rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                    rstd = arith.truncf(xrt_dtype, rstd_f32)

                    v_rstd = BroadcastOp(vecTy, rstd)
                    for j in for_(0, n, vector_size):
                        sub_row = subview(l1_row.result, [j], [vector_size], [1])
                        sub_w = subview(l1_weight_buf.result, [j], [vector_size], [1])
                        sub_out = subview(l1_out.result, [j], [vector_size], [1])
                        v_x = transfer_read(
                            vecTy, sub_row, [c0], identity_map, cst0, [True]
                        )
                        v_w = transfer_read(
                            vecTy, sub_w, [c0], identity_map, cst0, [True]
                        )
                        v_normed = arith.mulf(v_x, v_rstd)
                        v_weighted = arith.mulf(v_normed, v_w)
                        transfer_write(
                            None, v_weighted, sub_out, [c0], identity_map, [True]
                        )
                        yield_([])

                    dma_memcpy_nd(
                        l3_out,
                        l1_out,
                        dst_offsets=[row, 0],
                        dst_sizes=[1, n],
                        dst_strides=[n, 1],
                    )

                    DeallocOp(l1_row)
                    DeallocOp(l1_out)
                    DeallocOp(l1_weight_buf)
                    DeallocOp(l1_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RMSNorm 1D (decode shape, M=1) correctness + profile"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--N", type=int, default=2048, help="Cols (default: LLAMA emb_dim)"
    )
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    n = args.N
    print(f"RMSNorm 1D (decode): M=1, N={n}")
    module = build_module(n, bfloat16, args.vector_size)
    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(0)
    x_input = np.random.rand(1, n).astype(bfloat16)  # M=1
    weight = np.random.rand(n).astype(bfloat16)
    y_expected = rms_norm_reference(x_input, weight).reshape(n)
    x_flat = x_input.reshape(n)

    RTOL, ATOL, MIN_CORR = 5e-2, 5e-1, 0.99
    print(f"  Tolerances: rtol={RTOL}, atol={ATOL}, min_correlation={MIN_CORR}")

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="rms_norm_1d",
            runtime_loop_tiling_sizes=[4, 4],
        )
        artifact = backend.compile(module)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        out_buf = np.zeros((n,), dtype=bfloat16)
        inputs = [x_flat, weight, out_buf]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
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

        out = bos[-1].read(sizes[-1], 0).view(np.int16).view(bfloat16).reshape(n)
        out_f, ref_f = out.astype(np.float32), y_expected.astype(np.float32)
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
            instance_name="rms_norm_1d",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                module,
                inputs=[x_flat, weight],
                expected_outputs=[y_expected],
                rtol=RTOL,
                atol=ATOL,
                min_correlation=MIN_CORR,
            )
        )
