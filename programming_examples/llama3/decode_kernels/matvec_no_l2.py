#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GEMV bypassing L2: A and B go directly L3→L1 (no MemTile staging).
# Tests whether removing L2 overhead improves GEMV bandwidth.
#
# Compared to matrix_vector_multiplication/bf16/matvec.py:
# - A: L3→L1 directly (was L3→L2→L1)
# - B: L3→L1 directly (unchanged — already direct)
# - C: L1→L3 directly (was L1→L2→L3)

import argparse
import sys
import os
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
def build_module(m, k, tile_m, m_input, herd_m, np_dtype):
    assert m % (tile_m * herd_m) == 0
    assert tile_m % m_input == 0
    assert k % 64 == 0

    xrt_dtype = type_mapper(np_dtype)

    # L3 types
    memrefTyA = MemRefType.get([m, k], xrt_dtype)
    memrefTyB = MemRefType.get([k], xrt_dtype)
    memrefTyC = MemRefType.get([m], xrt_dtype)

    # L1 types only (no L2!)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1TyA = MemRefType.get([m_input, k], xrt_dtype, memory_space=l1_space)
    l1TyB = MemRefType.get([k], xrt_dtype, memory_space=l1_space)
    l1TyC = MemRefType.get([tile_m], xrt_dtype, memory_space=l1_space)

    # Check L1 budget
    l1_bytes = (m_input * k + k + tile_m) * 2
    assert l1_bytes <= 64 * 1024, f"L1 {l1_bytes} exceeds 64KB"

    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([T.i32(), T.i32(), T.i32(), l1TyA, l1TyB, l1TyC], []),
        visibility="private",
    )
    fill_func = FuncOp(
        "linalg_fill_bf16",
        ([xrt_dtype, l1TyC], []),
        visibility="private",
    )
    for f in [matvec_func, fill_func]:
        f.attributes["link_with"] = StringAttr.get("mv.o")
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyC)
    def matvec_no_l2(arg0, arg1, arg2):
        launch_size = [m // tile_m // herd_m, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(ivx, ivy, sx, sy, l3_a, l3_b, l3_c):

            @segment(name="seg", operands=[ivx, l3_a, l3_b, l3_c])
            def seg_body(ivx_s, s_a, s_b, s_c):
                launch_off_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_off = affine_apply(launch_off_map, [ivx_s])

                l1_a = AllocOp(l1TyA, [], [])
                l1_b = AllocOp(l1TyB, [], [])
                l1_c = AllocOp(l1TyC, [], [])

                # All data goes L3→L1 directly (no L2 alloc!)
                @herd(
                    name="herd_0",
                    sizes=[herd_m, 1],
                    operands=[l1_a, l1_b, l1_c, s_a, s_b, s_c, launch_off],
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_a,
                    _l1_b,
                    _l1_c,
                    _l3_a,
                    _l3_b,
                    _l3_c,
                    _launch_off,
                ):
                    # Zero-fill C
                    zero = ConstantOp(FloatAttr.get(xrt_dtype, 0), None)
                    CallOp(fill_func, [zero, _l1_c])

                    # Row base for this column: launch_off + _tx * tile_m
                    col_off_map = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(tile_m),
                                ),
                            )
                        ],
                    )
                    col_row_base = affine_apply(col_off_map, [_launch_off, _tx])

                    for j_m in range_(0, tile_m // m_input):
                        j_m_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        j_m_offset = affine_apply(j_m_map, [j_m])

                        # A: L3→L1 directly (col_row_base + j_m_offset rows)
                        a_row_map = AffineMap.get(
                            0,
                            2,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineSymbolExpr.get(1),
                                )
                            ],
                        )
                        a_row = affine_apply(a_row_map, [col_row_base, j_m_offset])
                        dma_memcpy_nd(
                            _l1_a,
                            _l3_a,
                            src_offsets=[a_row, 0],
                            src_sizes=[m_input, k],
                            src_strides=[k, 1],
                        )

                        # B: L3→L1 directly
                        dma_memcpy_nd(
                            _l1_b,
                            _l3_b,
                            src_offsets=[],
                            src_sizes=[k],
                            src_strides=[1],
                        )

                        # Kernel
                        row_off_i32 = arith.index_cast(T.i32(), j_m_offset)
                        CallOp(
                            matvec_func,
                            [
                                ConstantOp(IntegerAttr.get(T.i32(), m_input), None),
                                ConstantOp(IntegerAttr.get(T.i32(), k), None),
                                row_off_i32,
                                _l1_a,
                                _l1_b,
                                _l1_c,
                            ],
                        )
                        yield_([])

                    # C: L1→L3 directly
                    dma_memcpy_nd(
                        _l3_c,
                        _l1_c,
                        dst_offsets=[col_row_base],
                        dst_sizes=[tile_m],
                        dst_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get("mv.o")

                DeallocOp(l1_a)
                DeallocOp(l1_b)
                DeallocOp(l1_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMV bypassing L2")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=8)
    parser.add_argument("--m-input", type=int, default=4)
    parser.add_argument("--herd-m", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    module = build_module(
        args.m, args.k, args.tile_m, args.m_input, args.herd_m, bfloat16
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    A = (np.random.randn(args.m, args.k) * 4).astype(bfloat16)
    B = (np.random.randn(args.k) * 4).astype(bfloat16)
    ref = np.dot(A.astype(np.float32), B.astype(np.float32)).astype(bfloat16)

    if args.profile:
        import pyxrt as xrt
        import filelock

        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
        )

        artifact = backend.compile(module)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        invoker(A, B, np.zeros(args.m, dtype=bfloat16))  # warmup

        times = []
        for _ in range(args.iterations):
            c = np.zeros(args.m, dtype=bfloat16)
            t0 = time.perf_counter()
            results = invoker(A, B, c)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        npu = results[-1].astype(np.float32)
        corr = np.corrcoef(npu, ref.astype(np.float32))[0, 1]
        bw = (args.m * args.k + args.k + args.m) * 2 / (min(times) * 1e-6) / 1e9

        print(
            f"GEMV {args.m}x{args.k} no-L2: min={min(times):.0f}us avg={np.mean(times):.0f}us BW={bw:.1f}GB/s corr={corr:.6f}"
        )
        backend.unload()
    else:
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
        )
        exit(
            runner.run_test(
                module, inputs=[A, B], expected_outputs=[ref], rtol=0.04, atol=1e-3
            )
        )
