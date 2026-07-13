# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Forked from programming_examples/softmax/softmax.py.
#
# Adds a second L3 input `mask` (same shape as the score input). Each herd
# tile DMAs the corresponding mask tile alongside the data tile and calls
# masked_softmax_bf16(in, pos, mask, out), which computes softmax(in + mask)
# instead of softmax(in). This is the masked full-row softmax piece used to
# build non-causal (prefix/padding-mask) attention on top of the existing
# GEMM kernels: S = Q@K^T/sqrt(d); S += mask; P = row-softmax(S); O = P@V.
import argparse
from math import cos, sin, sqrt, exp

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
import ml_dtypes
from ml_dtypes import bfloat16

import numpy as np

np.random.seed(42)

range_ = for_

# Additive-mask "masked out" value: torch.finfo(bfloat16).min. This matches
# the real model, which does torch.where(mask, x, finfo.min) then softmax.
#
# Using the literal bf16 min (rather than some smaller ad hoc large-negative
# number) matters for a fully-masked row: bf16 has ~8 significand bits, so
# at this magnitude its ULU (~2.6e36) dwarfs any real attention score
# (O(1-10)). score + BF16_MIN therefore rounds to the *same* bit pattern
# for every position in a fully-masked row regardless of the underlying
# score -- i.e. addition behaves like torch.where's replacement -- so the
# row max equals that identical value, every (x - max) difference is
# exactly 0, and the row softmaxes to a uniform distribution instead of
# NaN. A smaller magnitude (e.g. -100) does NOT have this property: softmax
# is shift-invariant, so adding the *same* smaller constant to every entry
# of a fully-masked row would just reproduce softmax(raw scores), not a
# uniform row.
#
# This value overflows the masked_softmax_bf16 kernel's int16 fixed-point
# LUT index (SM_SCALE_FAC=8 -> only |value - row_max| <~ 128 is natively
# representable) for the *masked* entries of a partially-masked row, whose
# (value - row_max) is astronomically negative. masked_softmax.cc handles
# this with an explicit clamp (kExpDiffFloor = -120.0) applied to the
# quantizer input in exp_bf16 before the LUT lookup; see the comment there.
BF16_MIN = float(ml_dtypes.finfo(bfloat16).min)


@module_builder
def build_module(n, tile_n, herd_n, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    a_size = [n]
    out_size = a_size
    xrt_dtype_in = type_mapper(np_dtype_in)

    # L3 MemRefTypes
    l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1MemrefTy = MemRefType.get(
        shape=[tile_n],
        element_type=xrt_dtype_in,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    # Function declaration
    masked_softmax_func = FuncOp(
        "masked_softmax_bf16",
        ([l1MemrefTy, T.i32(), l1MemrefTy, l1MemrefTy], []),
        visibility="private",
    )
    for func in [masked_softmax_func]:
        func.attributes["link_with"] = StringAttr.get("masked_softmax.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
    def masked_softmax(arg0, arg1, arg2):
        @herd(
            name="herd_0",
            sizes=[1, herd_n],
            operands=[arg0, arg1, arg2],
        )
        def herd_body(
            _tx,
            _ty,
            _sx,
            _sy,
            _l3_a,
            _l3_mask,
            _l3_c,
        ):
            l1_a_data = AllocOp(l1MemrefTy, [], [])
            l1_mask_data = AllocOp(l1MemrefTy, [], [])
            l1_out_data = AllocOp(l1MemrefTy, [], [])

            for t in range_(0, n, tile_n * herd_n):

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
                offset = affine_apply(offset_map, [t, _ty])

                dma_memcpy_nd(
                    l1_a_data,
                    _l3_a,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                dma_memcpy_nd(
                    l1_mask_data,
                    _l3_mask,
                    src_offsets=[
                        offset,
                    ],
                    src_sizes=[tile_n],
                    src_strides=[1],
                )
                const_pos = ConstantOp(IntegerAttr.get(T.i32(), tile_n - 1), None)
                masked_softmax_call = CallOp(
                    masked_softmax_func,
                    [l1_a_data, const_pos, l1_mask_data, l1_out_data],
                )
                dma_memcpy_nd(
                    _l3_c,
                    l1_out_data,
                    dst_offsets=[
                        offset,
                    ],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                DeallocOp(l1_a_data)
                DeallocOp(l1_mask_data)
                DeallocOp(l1_out_data)

                yield_([])

        herd_body.attributes["link_with"] = StringAttr.get("masked_softmax.o")


if __name__ == "__main__":
    # Default values. SmolVLA attention scores are per-head (256, 256): 256
    # rows (seq, padded from 241), softmax over the last dim of width 256.
    ROWS = 256
    TILE_N = 256
    N = ROWS * TILE_N
    HERD_N = 4
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the masked_softmax example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.n,
        args.tile_n,
        args.herd_n,
        INPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Masked softmax: scores + additive mask, then row-softmax over the last
    # dim (width = args.tile_n, i.e. the reduction axis).
    num_tiles = args.n // args.tile_n
    scores = np.random.randn(num_tiles, args.tile_n).astype(np.float32)

    # Build a random 0 / BF16_MIN additive mask, with at least one fully
    # masked row to exercise the uniform-row (no-NaN) case.
    keep = np.random.rand(num_tiles, args.tile_n) > 0.3  # ~70% attended
    mask_f32 = np.where(keep, 0.0, BF16_MIN).astype(np.float32)
    mask_f32[0, :] = BF16_MIN  # row 0: fully masked

    inputs_bf16 = scores.astype(INPUT_DATATYPE)
    mask_bf16 = mask_f32.astype(INPUT_DATATYPE)

    # Reference: computed in float32 on (scores + mask), matching the
    # bf16-rounded values actually seen on device (mask/scores are rounded
    # to bf16 first, then added/softmaxed in float32 as a numerically-stable
    # stand-in for the LUT-based bf16 kernel).
    combined = inputs_bf16.astype(np.float32) + mask_bf16.astype(np.float32)
    outputs = np.zeros(shape=(num_tiles, args.tile_n), dtype=INPUT_DATATYPE)
    for j in range(num_tiles):
        row = combined[j]
        max_val = np.max(row)
        exp_row = np.zeros(args.tile_n, dtype=np.float64)
        for i in range(args.tile_n):
            exp_row[i] = exp(float(row[i]) - float(max_val))
        sum_val = np.sum(exp_row)
        outputs[j] = (exp_row / sum_val).astype(INPUT_DATATYPE)

    # Sanity check the reference itself: a fully-masked row must be uniform
    # (not NaN).
    assert not np.any(np.isnan(outputs)), "Reference produced NaN"
    uniform_val = 1.0 / args.tile_n
    assert np.allclose(
        outputs[0].astype(np.float32), uniform_val, rtol=1e-2, atol=1e-2
    ), "Fully-masked row did not reduce to a uniform distribution in the reference"

    if args.compile_mode == "compile-and-run":

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="masked_softmax",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs_bf16, mask_bf16],
                expected_outputs=[outputs],
                rtol=1.6e-2,
                atol=3e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
