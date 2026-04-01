#!/usr/bin/env python3
"""Minimal reproducer: airrt-to-npu fails when mixing herd-only and segment
launches in a multi-launch ELF.

Bug: `identifyLaunchRegions()` in AIRRtToNpuPass.cpp only looks for
`airrt::SegmentLoadOp`. If a launch has a bare `air.herd` (no `air.segment`),
it lowers to `airrt.herd_load` which is silently skipped. When OTHER launches
have `segment_load`, the fallback path (which handles both) is never reached.
The bare-herd launch's DMA ops get orphaned outside any `aie.device`, and
`DmaToNpuPattern` fails with "failed to legalize operation 'airrt.dma_memcpy_nd'".

Test 1: Two launches, BOTH with air.segment → PASS
Test 2: Two launches, one with air.segment, one with bare air.herd → FAIL
Test 3: Two launches, BOTH with bare air.herd → FAIL (fallback also broken for ELF)

Usage:
    python3 repro_herd_load_bug.py          # runs all tests
    python3 repro_herd_load_bug.py --print  # print MLIR for failing case
"""

import argparse
import sys

from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend

range_ = for_


def _build_add_launch(
    xrt_dtype, n, tile_n, herd_x, vector_size, use_segment, name="add"
):
    """Build an eltwise add air.launch, optionally with air.segment."""
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
                    AffineSymbolExpr.get(1),
                    AffineConstantExpr.get(tile_n),
                ),
            )
        ],
    )

    def _herd_body(ha, hb, ho, tx, ty):
        l1a = AllocOp(l1_ty, [], [])
        l1b = AllocOp(l1_ty, [], [])
        l1o = AllocOp(l1_ty, [], [])
        c0 = arith.ConstantOp.create_index(0)
        cst = arith.ConstantOp(xrt_dtype, 0.0)
        for iv in range_(0, chunk, tile_n):
            off = affine_apply(omap, [iv, tx])
            dma_memcpy_nd(
                l1a, ha, src_offsets=[off], src_sizes=[tile_n], src_strides=[1]
            )
            dma_memcpy_nd(
                l1b, hb, src_offsets=[off], src_sizes=[tile_n], src_strides=[1]
            )
            for j in range_(0, tile_n, vector_size):
                sa = subview(l1a.result, [j], [vector_size], [1])
                sb = subview(l1b.result, [j], [vector_size], [1])
                so = subview(l1o.result, [j], [vector_size], [1])
                va = transfer_read(vec_ty, sa, [c0], imap, cst, [True])
                vb = transfer_read(vec_ty, sb, [c0], imap, cst, [True])
                transfer_write(None, arith.addf(va, vb), so, [c0], imap, [True])
                yield_([])
            dma_memcpy_nd(
                ho, l1o, dst_offsets=[off], dst_sizes=[tile_n], dst_strides=[1]
            )
            yield_([])
        DeallocOp(l1a)
        DeallocOp(l1b)
        DeallocOp(l1o)

    def builder(a, b, out):
        @launch(operands=[a, b, out])
        def the_launch(la, lb, lo):
            if use_segment:

                @segment(name=f"{name}_seg", operands=[la, lb, lo])
                def seg(sa, sb, so):
                    @herd(name=f"{name}_herd", sizes=[herd_x, 1], operands=[sa, sb, so])
                    def h(tx, ty, sx, sy, ha, hb, ho):
                        _herd_body(ha, hb, ho, tx, ty)

            else:
                # Bare herd inside launch — no segment wrapper
                @herd(name=f"{name}_herd", sizes=[herd_x, 1], operands=[la, lb, lo])
                def h(tx, ty, sx, sy, ha, hb, ho):
                    _herd_body(ha, hb, ho, tx, ty)

    return builder


from air.dialects.affine import apply as affine_apply


def build_two_launches(n, use_segment_1, use_segment_2):
    """Build a module with two eltwise-add launches."""
    from air.backend.xrt_runner import type_mapper

    @module_builder
    def mod():
        xrt = type_mapper(bfloat16)
        l3_ty = MemRefType.get([n], xrt)
        b1 = _build_add_launch(xrt, n, 128, 8, 16, use_segment_1, name="add1")
        b2 = _build_add_launch(xrt, n, 128, 8, 16, use_segment_2, name="add2")

        @FuncOp.from_py_func(l3_ty, l3_ty, l3_ty, l3_ty, l3_ty, l3_ty)
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
        return False, lines[0][:150] if lines else str(e)[:150]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print failing MLIR")
    args = parser.parse_args()

    N = 1024  # Small size

    if args.print:
        print("=== FAILING: Launch 1 has segment, Launch 2 has bare herd ===")
        print(build_two_launches(N, use_segment_1=True, use_segment_2=False))
        sys.exit(0)

    print("Test 1: Both launches have air.segment (should PASS)")
    ok, msg = compile_test("both_seg", build_two_launches(N, True, True))
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

    print("Test 2: Launch 1 has segment, Launch 2 has bare herd (should FAIL)")
    ok2, msg2 = compile_test("mixed", build_two_launches(N, True, False))
    print(f"  {'PASS' if ok2 else 'FAIL'}: {msg2}")

    print("Test 3: Both launches have bare herd (also FAIL in ELF mode)")
    ok3, msg3 = compile_test("both_herd", build_two_launches(N, False, False))
    print(f"  {'PASS' if ok3 else 'FAIL'}: {msg3}")

    if not ok2 and ok:
        print("\nReproduced! Bare air.herd (without air.segment) fails in")
        print("multi-launch ELF. Wrapping herd in segment fixes the issue.")
    elif ok2:
        print("\nBug not reproduced — all tests passed.")
