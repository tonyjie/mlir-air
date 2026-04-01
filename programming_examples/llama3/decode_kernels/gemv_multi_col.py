#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-column GEMV for decode: C[M] = A[M,K] @ B[K]

Distributes M rows across num_cols AIE columns. Each column computes
M/num_cols output rows independently. B vector is broadcast to all columns.

Architecture:
  - Launch grid: [M // tile_m_total, 1] where tile_m_total = num_cols * tile_m_per_col
  - Herd: [1, num_cols] — columns work in parallel
  - Each column: streams m_input rows of A from L2, accumulates into C
  - B: loaded once into L1 per column (broadcast)

Uses mv.o external kernel (same as matrix_vector_multiplication/bf16).

Usage:
    python3 gemv_multi_col.py --m 2048 --k 2048 --num-cols 4
    python3 gemv_multi_col.py --m 8192 --k 2048 --num-cols 4 --profile
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(m, k, num_cols, m_input, np_dtype):
    """Build multi-column GEMV: C[M] = A[M,K] @ B[K].

    Args:
        m: Output rows (M dimension).
        k: Reduction dimension (vector length).
        num_cols: Number of AIE columns (herd width).
        m_input: Rows per kernel call.
        np_dtype: Data type (bfloat16).
    """
    assert m % num_cols == 0, f"M ({m}) must be divisible by num_cols ({num_cols})"
    m_per_col = m // num_cols
    assert m_per_col % m_input == 0
    assert k % 64 == 0, f"K ({k}) must be divisible by 64"

    xrt_dtype = type_mapper(np_dtype)

    # L3 types
    l3_a_ty = MemRefType.get([m * k], xrt_dtype)  # Flat A matrix
    l3_b_ty = MemRefType.get([k], xrt_dtype)  # B vector
    l3_c_ty = MemRefType.get([m], xrt_dtype)  # C output

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_a_ty = MemRefType.get([m_input * k], xrt_dtype, memory_space=l1_space)
    l1_b_ty = MemRefType.get([k], xrt_dtype, memory_space=l1_space)
    l1_c_ty = MemRefType.get([m_per_col], xrt_dtype, memory_space=l1_space)

    # Check L1 budget
    l1_total = (m_input * k + k + m_per_col) * 2  # bf16 = 2 bytes
    assert l1_total <= 64 * 1024, (
        f"L1 budget exceeded: {l1_total} bytes > 64KB. "
        f"Reduce m_per_col (fewer cols or smaller M) or m_input."
    )

    # External kernels (from mv.o)
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([T.i32(), T.i32(), T.i32(), l1_a_ty, l1_b_ty, l1_c_ty], []),
        visibility="private",
    )
    linalg_fill_func = FuncOp(
        "linalg_fill_bf16",
        ([xrt_dtype, l1_c_ty], []),
        visibility="private",
    )
    for func in [matvec_func, linalg_fill_func]:
        func.attributes["link_with"] = StringAttr.get("mv.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_a_ty, l3_b_ty, l3_c_ty)
    def matvec_multi_col(arg_a, arg_b, arg_c):
        @launch(operands=[arg_a, arg_b, arg_c])
        def launch_body(l_a, l_b, l_c):
            @segment(
                name="gemv_seg",
                operands=[l_a, l_b, l_c],
            )
            def seg_body(s_a, s_b, s_c):
                # Each column handles m_per_col rows
                @herd(
                    name="gemv_herd",
                    sizes=[1, num_cols],
                    operands=[s_a, s_b, s_c],
                )
                def herd_body(_tx, _ty, _sx, _sy, h_a, h_b, h_c):
                    l1_a = AllocOp(l1_a_ty, [], [])
                    l1_b = AllocOp(l1_b_ty, [], [])
                    l1_c = AllocOp(l1_c_ty, [], [])

                    # Load B vector (same data to all cols)
                    # Use a dummy tile-dependent offset (0 * _ty = 0) to avoid
                    # the broadcast DMA compiler bug (stride=0 in aie.dma_bd)
                    dummy_zero_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(0),
                            )
                        ],
                    )
                    b_offset = affine_apply(dummy_zero_map, [_ty])
                    dma_memcpy_nd(
                        l1_b,
                        h_b,
                        src_offsets=[b_offset],
                        src_sizes=[k],
                        src_strides=[1],
                    )

                    # Zero-fill output
                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype, 0), None)
                    CallOp(linalg_fill_func, [zero_const, l1_c])

                    # Col base offset: _ty * m_per_col * k
                    col_base_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(m_per_col * k),
                            )
                        ],
                    )
                    col_a_base = affine_apply(col_base_map, [_ty])

                    # Process m_per_col rows, m_input at a time
                    for j_m in range_(0, m_per_col // m_input):
                        # A offset: col_base + j_m * m_input * k
                        row_off_map = AffineMap.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(1),
                                        AffineConstantExpr.get(m_input * k),
                                    ),
                                )
                            ],
                        )
                        a_offset = affine_apply(row_off_map, [col_a_base, j_m])

                        # DMA A rows: m_input × k elements
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[a_offset],
                            src_sizes=[m_input * k],
                            src_strides=[1],
                        )

                        # Row offset within output buffer
                        row_off_i32_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        row_offset = affine_apply(row_off_i32_map, [j_m])
                        row_offset_i32 = arith.index_cast(T.i32(), row_offset)
                        m_const = ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                        k_const = ConstantOp(IntegerAttr.get(T.i32(), k), None)

                        CallOp(
                            matvec_func,
                            [m_const, k_const, row_offset_i32, l1_a, l1_b, l1_c],
                        )
                        yield_([])

                    # Write output: col _ty writes to c[_ty*m_per_col : (_ty+1)*m_per_col]
                    col_c_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(m_per_col),
                            )
                        ],
                    )
                    c_offset = affine_apply(col_c_map, [_ty])
                    dma_memcpy_nd(
                        h_c,
                        l1_c,
                        dst_offsets=[c_offset],
                        dst_sizes=[m_per_col],
                        dst_strides=[1],
                    )

                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-column GEMV")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--num-cols", type=int, default=4)
    parser.add_argument("--m-input", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    print(
        f"Multi-col GEMV: M={args.m}, K={args.k}, cols={args.num_cols}, m_input={args.m_input}"
    )

    module = build_module(args.m, args.k, args.num_cols, args.m_input, bfloat16)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    input_a = (np.random.randn(args.m, args.k) * 4).astype(bfloat16)
    input_b = (np.random.randn(args.k) * 4).astype(bfloat16)
    output_ref = np.dot(input_a.astype(np.float32), input_b.astype(np.float32)).astype(
        bfloat16
    )

    if args.profile:
        import pyxrt as xrt
        import filelock

        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="xclbin",
            instance_name="gemv_multi",
        )
        artifact = backend.compile(module)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        # Warmup
        invoker(input_a.flatten(), input_b, np.zeros(args.m, dtype=bfloat16))

        times = []
        for _ in range(args.iterations):
            c_buf = np.zeros(args.m, dtype=bfloat16)
            t0 = time.perf_counter()
            results = invoker(input_a.flatten(), input_b, c_buf)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        npu = results[-1].astype(np.float32)
        corr = np.corrcoef(npu, output_ref.astype(np.float32))[0, 1]
        bw = args.m * args.k * 2 / (np.mean(times) * 1e-6) / 1e9

        print(f"Avg: {np.mean(times):.0f}µs, Min: {np.min(times):.0f}µs")
        print(f"Bandwidth: {bw:.1f} GB/s, Corr: {corr:.6f}")
        print(f"{'PASS' if corr > 0.99 else 'FAIL'}")
        backend.unload()
    else:
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            instance_name="gemv_multi",
        )
        exit(
            runner.run_test(
                module,
                inputs=[input_a.flatten(), input_b],
                expected_outputs=[output_ref],
                rtol=0.04,
                atol=1e-3,
            )
        )
