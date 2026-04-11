# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4 Matrix-vector multiplication (GEMV): C[M] = dequant(A_q4[M,K]) @ B[K]
#
# Interleaved Q4 weight format: each block of 32 values is stored as
# [16B packed Q4 | 2B scale (bf16) | 2B min (bf16)] = 20 bytes per block.
# Row bytes = K/32 * 20 = K * 5/8 (3.2x smaller than BF16).
#
# Same DMA pattern as bf16/matvec.py (L2 staging for A, L3→L1 for B).
# Only the weight type changes: memref<M×K×bf16> → memref<M×row_bytes×ui8>.

import argparse
import numpy as np
from ml_dtypes import bfloat16

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

BLOCK_SIZE = 32
BLOCK_BYTES = 20  # 16B packed Q4 + 2B scale + 2B min


def row_bytes(k):
    """Bytes per row in interleaved Q4 format."""
    return (k // BLOCK_SIZE) * BLOCK_BYTES


@module_builder
def build_module(m, k, tile_m, m_input, herd_m, np_dtype_out=bfloat16):
    """Build Q4 GEMV module with single interleaved weight buffer.

    Same 3-arg pattern as BF16 GEMV: (weight, input_vec, output_vec).
    Weight type is memref<M × row_bytes × ui8> instead of memref<M × K × bf16>.
    """
    assert m % (tile_m * herd_m) == 0
    assert tile_m % m_input == 0
    assert k % BLOCK_SIZE == 0

    rb = row_bytes(k)  # bytes per row

    xrt_bf16 = type_mapper(bfloat16)
    xrt_out = type_mapper(np_dtype_out)
    xrt_u8 = IntegerType.get_signless(8)

    # L3 types — weight is 1D flat (M*rb bytes) to avoid 2D memref BO issues
    memrefTyA = MemRefType.get([m * rb], xrt_u8)  # Q4 interleaved weight (flat)
    memrefTyB = MemRefType.get([k], xrt_bf16)
    memrefTyC = MemRefType.get([m], xrt_out)

    # L2 types
    l2_mem = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get([herd_m, tile_m, rb], xrt_u8, memory_space=l2_mem)
    l2MemrefTyC = MemRefType.get([herd_m, tile_m], xrt_out, memory_space=l2_mem)

    # L1 types
    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get([m_input, rb], xrt_u8, memory_space=l1_mem)
    l1MemrefTyB = MemRefType.get([k], xrt_bf16, memory_space=l1_mem)
    l1MemrefTyC = MemRefType.get([tile_m], xrt_out, memory_space=l1_mem)

    # External kernel — same arg count as BF16: (m, k, offset, A, B, C)
    q4_func = FuncOp(
        "q4_matvec_bf16",
        ([T.i32(), T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyC], []),
        visibility="private",
    )
    q4_func.attributes["link_with"] = StringAttr.get("mv_q4.o")
    q4_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyC)
    def q4_matvec(arg0, arg1, arg2):

        launch_size = [m // tile_m // herd_m, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a,
            l3_b,
            l3_c,
        ):
            @segment(
                name="q4_matvec_seg",
                operands=[launch_ivx, l3_a, l3_b, l3_c],
            )
            def segment_body(launch_ivx_s, l3_a_s, l3_b_s, l3_c_s):
                launch_ivx_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_offset_m = affine_apply(launch_ivx_map, [launch_ivx_s])

                l2_a = AllocOp(l2MemrefTyA, [], [])
                l2_c = AllocOp(l2MemrefTyC, [], [])
                l1_a = AllocOp(l1MemrefTyA, [], [])
                l1_b = AllocOp(l1MemrefTyB, [], [])
                l1_c = AllocOp(l1MemrefTyC, [], [])

                # L3→L2: Q4 weight tile (same pattern as BF16, just rb instead of k)
                dma_memcpy_nd(
                    l2_a,
                    l3_a_s,
                    src_offsets=[0, launch_offset_m, 0],
                    src_sizes=[herd_m, tile_m, rb],
                    src_strides=[tile_m * rb, rb, 1],
                )

                @herd(
                    name="herd_0",
                    sizes=[herd_m, 1],
                    operands=[l1_a, l1_b, l1_c, l2_a, l3_b_s, l2_c],
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_a,
                    _l1_b,
                    _l1_c,
                    _l2_a,
                    _l3_b,
                    _l2_c,
                ):
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

                        # L3→L1: B (direct)
                        dma_memcpy_nd(
                            _l1_b,
                            _l3_b,
                            src_offsets=[],
                            src_sizes=[k],
                            src_strides=[1],
                        )

                        # L2→L1: Q4 weight rows
                        dma_memcpy_nd(
                            _l1_a,
                            _l2_a,
                            src_offsets=[_tx, j_m_offset, 0],
                            src_sizes=[1, m_input, rb],
                            src_strides=[tile_m * rb, rb, 1],
                        )

                        # Kernel call
                        row_offset_i32 = arith.index_cast(T.i32(), j_m_offset)
                        m_const = ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                        k_const = ConstantOp(IntegerAttr.get(T.i32(), k), None)

                        CallOp(
                            q4_func,
                            [
                                m_const,
                                k_const,
                                row_offset_i32,
                                _l1_a,
                                _l1_b,
                                _l1_c,
                            ],
                        )

                        yield_([])

                    # L1→L2: C
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c,
                        dst_offsets=[_tx, 0],
                        dst_sizes=[1, tile_m],
                        dst_strides=[tile_m, 1],
                        src_offsets=[],
                        src_sizes=[tile_m],
                        src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get("mv_q4.o")

                # L2→L3: C
                dma_memcpy_nd(
                    l3_c_s,
                    l2_c,
                    dst_offsets=[launch_offset_m],
                    dst_sizes=[herd_m * tile_m],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_m, tile_m],
                    src_strides=[tile_m, 1],
                )

                DeallocOp(l2_a)
                DeallocOp(l2_c)
                DeallocOp(l1_a)
                DeallocOp(l1_b)
                DeallocOp(l1_c)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../")

    parser = argparse.ArgumentParser(description="Q4 GEMV test")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=8)
    parser.add_argument("--m-input", type=int, default=4)
    parser.add_argument("--herd-m", type=int, default=8)
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    args = parser.parse_args()

    M, K = args.m, args.k
    TILE_M, M_INPUT, HERD_M = args.tile_m, args.m_input, args.herd_m

    print(f"Q4 GEMV: M={M}, K={K}, tile_m={TILE_M}, m_input={M_INPUT}, herd_m={HERD_M}")

    module = build_module(M, K, TILE_M, M_INPUT, HERD_M)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    from llama3.kernel_builder.quantize import (
        pack_q4_interleaved,
        q4_interleaved_dequant_reference,
    )

    np.random.seed(42)
    w_bf16 = np.random.uniform(-1, 1, (M, K)).astype(bfloat16)
    x_bf16 = np.random.uniform(-1, 1, K).astype(bfloat16)

    # Pack to interleaved Q4 format
    w_packed = pack_q4_interleaved(w_bf16)
    print(
        f"  Weight: {w_bf16.shape} BF16 ({w_bf16.nbytes}B) → {w_packed.shape} Q4 ({w_packed.nbytes}B)"
    )

    # CPU reference
    w_dequant = q4_interleaved_dequant_reference(w_packed, M, K)
    y_ref = (w_dequant @ x_bf16.astype(np.float32)).astype(bfloat16)
    print(f"  y_ref sample: {y_ref[:4]}")

    c_out = np.zeros(M, dtype=bfloat16)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="q4_matvec",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="q4_matvec",
    )
    exit_code = runner.run_test(
        module,
        inputs=[w_packed.flatten(), x_bf16],
        expected_outputs=[y_ref],
        rtol=0.05,
        atol=0.1,
    )
    sys.exit(exit_code)
