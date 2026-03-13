# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone SwiGLU Activation Kernel

Element-wise SwiGLU: output[i] = SiLU(gate[i]) * up[i]
where SiLU(x) = x * sigmoid(x)

Uses an external C++ kernel (swiglu_activation.cc) compiled with Peano.
The kernel processes data in tiles using a 1x2 herd (2 AIE tiles).
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in):
    xrt_dtype = type_mapper(np_dtype_in)
    num_tiles = 2
    assert (
        n % (tile_n * num_tiles) == 0
    ), f"n ({n}) must be divisible by tile_n * num_tiles ({tile_n * num_tiles})"

    # L3 types
    l3MemrefTy = MemRefType.get([n], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    # External kernel declaration
    swiglu_func = FuncOp(
        "swiglu_bf16",
        ([l1MemrefTy, l1MemrefTy, l1MemrefTy, T.i32()], []),
        visibility="private",
    )
    swiglu_func.attributes["link_with"] = StringAttr.get("swiglu_activation.o")
    swiglu_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3MemrefTy, l3MemrefTy, l3MemrefTy)
    def swiglu_activation(arg0, arg1, arg2):
        # arg0 = gate [n], arg1 = up [n], arg2 = output [n]

        @herd(name="herd_0", sizes=[1, num_tiles], operands=[arg0, arg1, arg2])
        def herd_body(_tx, _ty, _sx, _sy, l3_gate, l3_up, l3_out):
            l1_gate = AllocOp(l1MemrefTy, [], [])
            l1_up = AllocOp(l1MemrefTy, [], [])
            l1_out = AllocOp(l1MemrefTy, [], [])

            tile_n_i32 = ConstantOp(T.i32(), tile_n)

            for loop_iv in range_(0, n, tile_n * num_tiles):
                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(tile_n),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [loop_iv, _ty])

                dma_memcpy_nd(
                    l1_gate,
                    l3_gate,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_up,
                    l3_up,
                    src_offsets=[offset],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )

                CallOp(swiglu_func, [l1_gate, l1_up, l1_out, tile_n_i32])

                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[offset],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                yield_([])

            DeallocOp(l1_gate)
            DeallocOp(l1_up)
            DeallocOp(l1_out)

        herd_body.attributes["link_with"] = StringAttr.get("swiglu_activation.o")


def silu_reference(x):
    """Reference SiLU implementation in F32."""
    x_f32 = x.astype(np.float32)
    return x_f32 * (1.0 / (1.0 + np.exp(-x_f32)))


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="swiglu_activation.py",
        description="Builds, runs, and tests the standalone SwiGLU activation kernel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    gate = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)
    up = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    # Reference: SiLU(gate) * up
    silu_gate = silu_reference(gate)
    expected = (silu_gate * up.astype(np.float32)).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="swiglu_activation",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[gate, up],
                expected_outputs=[expected],
                rtol=5e-2,
                atol=5e-1,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
