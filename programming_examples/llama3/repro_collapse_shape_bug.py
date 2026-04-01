#!/usr/bin/env python3
"""Minimal reproducer: airrt-to-npu fails with two collapse_shape launches.

Bug: When a multi-launch ELF has TWO air.launch ops that each contain
memref.collapse_shape (2D→1D), the airrt-to-npu pass fails with:
  "failed to legalize operation 'airrt.dma_memcpy_nd'"

Root cause: An earlier pass hoists collapse_shape out of the air.launch body.
For one launch, canonicalize folds it away. For the second, it persists as
a func-level SSA value that airrt-to-npu can't trace to a function argument.

Single collapse_shape launch: PASS
Two collapse_shape launches: FAIL (even at tiny scale)

Usage:
    python3 repro_collapse_shape_bug.py          # runs both tests
    python3 repro_collapse_shape_bug.py --print   # print the failing MLIR
"""

import argparse
import sys

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def _build_add_2d_launch(
    l3_2d_ty, l3_1d_ty, xrt_dtype, herd_name, n, tile_n, herd_x, vector_size
):
    """Build an eltwise add air.launch that accepts 2D memrefs and
    does collapse_shape inside the launch body."""
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
    vec_ty = VectorType.get([vector_size], xrt_dtype)
    imap = AffineMapAttr.get(AffineMap.get_identity(1))
    chunk = n // herd_x
    omap = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1), AffineConstantExpr.get(tile_n)
                ),
            )
        ],
    )

    def builder(a_2d, b_2d, out_1d):
        @launch(operands=[a_2d, b_2d, out_1d])
        def add_launch(la, lb, lo):
            # collapse_shape INSIDE the launch body
            af = memref_collapse_shape(l3_1d_ty, la, [[0, 1]])
            bf = memref_collapse_shape(l3_1d_ty, lb, [[0, 1]])

            @segment(name=f"{herd_name}_seg", operands=[af, bf, lo])
            def seg(sa, sb, so):
                @herd(name=herd_name, sizes=[herd_x, 1], operands=[sa, sb, so])
                def h(tx, ty, sx, sy, ha, hb, ho):
                    l1a = AllocOp(l1_ty, [], [])
                    l1b = AllocOp(l1_ty, [], [])
                    l1o = AllocOp(l1_ty, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst = arith.ConstantOp(xrt_dtype, 0.0)
                    for iv in range_(0, chunk, tile_n):
                        off = affine_apply(omap, [iv, tx])
                        dma_memcpy_nd(
                            l1a,
                            ha,
                            src_offsets=[off],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1b,
                            hb,
                            src_offsets=[off],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in range_(0, tile_n, vector_size):
                            sa_ = subview(l1a.result, [j], [vector_size], [1])
                            sb_ = subview(l1b.result, [j], [vector_size], [1])
                            so_ = subview(l1o.result, [j], [vector_size], [1])
                            va = transfer_read(vec_ty, sa_, [c0], imap, cst, [True])
                            vb = transfer_read(vec_ty, sb_, [c0], imap, cst, [True])
                            transfer_write(
                                None, arith.addf(va, vb), so_, [c0], imap, [True]
                            )
                            yield_([])
                        dma_memcpy_nd(
                            ho,
                            l1o,
                            dst_offsets=[off],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(l1a)
                    DeallocOp(l1b)
                    DeallocOp(l1o)

    return builder


def build_single_collapse(M=128, N=128):
    """One launch with collapse_shape — should PASS."""

    @module_builder
    def mod():
        xrt = type_mapper(bfloat16)
        n = M * N
        l3_2d = MemRefType.get([M, N], xrt)
        l3_1d = MemRefType.get([n], xrt)
        builder = _build_add_2d_launch(l3_2d, l3_1d, xrt, "add1", n, N, 8, 16)

        @FuncOp.from_py_func(l3_2d, l3_2d, l3_1d)
        def test(a, b, out):
            builder(a, b, out)

    return mod()


def build_two_collapse(M=128, N=128):
    """Two launches, BOTH with collapse_shape — may FAIL."""

    @module_builder
    def mod():
        xrt = type_mapper(bfloat16)
        n = M * N
        l3_2d = MemRefType.get([M, N], xrt)
        l3_1d = MemRefType.get([n], xrt)
        b1 = _build_add_2d_launch(l3_2d, l3_1d, xrt, "add1", n, N, 8, 16)
        b2 = _build_add_2d_launch(l3_2d, l3_1d, xrt, "add2", n, N, 8, 16)

        @FuncOp.from_py_func(l3_2d, l3_2d, l3_1d, l3_2d, l3_2d, l3_1d)
        def test(a1, b1_arg, out1, a2, b2_arg, out2):
            b1(a1, b1_arg, out1)
            b2(a2, b2_arg, out2)

    return mod()


def compile_test(name, module):
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name=name,
    )
    try:
        artifact = backend.compile(module)
        backend.unload()
        return True, "OK"
    except Exception as e:
        backend.unload()
        lines = [l for l in str(e).split("\n") if "error:" in l.lower()]
        return False, lines[0][:120] if lines else str(e)[:120]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print MLIR only")
    args = parser.parse_args()

    if args.print:
        print("=== Single collapse_shape (should PASS) ===")
        print(build_single_collapse())
        print("\n=== Two collapse_shapes (may FAIL) ===")
        print(build_two_collapse())
        sys.exit(0)

    print("Test 1: Single launch with collapse_shape inside")
    ok, msg = compile_test("single", build_single_collapse())
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

    print("Test 2: Two launches, both with collapse_shape inside")
    ok, msg = compile_test("double", build_two_collapse())
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

    if ok:
        print("\nBoth passed — bug may not reproduce with this minimal case.")
        print("Try with the full 6-launch ffn_full_multi.py instead.")
    else:
        print("\nReproduced! The second collapse_shape launch causes the failure.")
