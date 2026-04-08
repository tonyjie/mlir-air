# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE (LUT-based) with Tiled L2 Staging for 2D Herds

Uses L2 (MemTile) staging to enable 2D herds (e.g. [8,4] = 32 tiles).

Key architecture insight (matching GEMM's L2 pattern):
  L2 buffer shape: [herd_x, col_chunk]
    - Dim 0 (herd_x) → maps to MemTile selection (one MemTile per column, free)
    - Dim 1 (col_chunk) → data for all herd_y tiles in this column
    - col_chunk = herd_y * rows_per_tile * embed_dim

  DDR → L2: bulk transfer, no _tx/_ty dependency → 1 BD at shim
  L2 → L1: src_offsets=[_tx, ty_offset]
    - _tx indexes dim 0 → MemTile routing (no BD cost)
    - ty_offset indexes within column's data using _ty → herd_y BDs per MemTile

This avoids the 32-BD-per-buffer problem that occurs when both _tx and _ty
are in the same DMA offset dimension.

Outer tiling loop at segment level processes rows in batches that fit MemTile.
"""

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


@module_builder
def build_module(
    seq_len, embed_dim, np_dtype_in, herd_x=8, herd_y=4, rows_per_batch=None
):
    xrt_dtype = type_mapper(np_dtype_in)
    total = seq_len * embed_dim
    total_tiles = herd_x * herd_y
    assert embed_dim % 16 == 0, "embed_dim must be divisible by 16"
    assert (
        seq_len % total_tiles == 0
    ), f"seq_len ({seq_len}) must be divisible by total tiles ({total_tiles})"

    # Auto-compute rows_per_batch to fit in MemTile (~256KB per column)
    # Each column stores: herd_y * rows_per_tile rows × embed_dim × 2 bytes × 3 buffers
    # Budget per column per buffer: ~80KB → 80*1024 / (embed_dim * 2) rows
    if rows_per_batch is None:
        max_rows_per_col = 80 * 1024 // (embed_dim * 2)  # ~640 for dim=64
        rows_per_tile_max = max_rows_per_col // herd_y
        rows_per_batch = rows_per_tile_max * total_tiles
        rows_per_batch = max(rows_per_batch, total_tiles)  # at least 1 row per tile
        # Round down to nearest multiple of total_tiles that divides seq_len
        while seq_len % rows_per_batch != 0 and rows_per_batch > total_tiles:
            rows_per_batch -= total_tiles
        rows_per_batch = min(rows_per_batch, seq_len)

    assert (
        seq_len % rows_per_batch == 0
    ), f"seq_len ({seq_len}) must be divisible by rows_per_batch ({rows_per_batch})"
    assert (
        rows_per_batch % total_tiles == 0
    ), f"rows_per_batch ({rows_per_batch}) must be divisible by total_tiles ({total_tiles})"

    rows_per_tile = rows_per_batch // total_tiles
    # Per-column data: herd_y tiles × rows_per_tile rows × embed_dim elements
    col_chunk = herd_y * rows_per_tile * embed_dim
    batch_size_elems = rows_per_batch * embed_dim

    print(
        f"  L2 config: rows_per_batch={rows_per_batch}, rows_per_tile={rows_per_tile}, "
        f"col_chunk={col_chunk} elems ({col_chunk * 2 // 1024}KB), "
        f"num_batches={seq_len // rows_per_batch}"
    )

    # L3 types
    l3DataTy = MemRefType.get([total], xrt_dtype)

    # L2 types: [herd_x, col_chunk] — dim 0 maps to MemTile, dim 1 is per-column data
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2DataTy = MemRefType.get([herd_x, col_chunk], xrt_dtype, memory_space=l2_mem_space)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get(
        shape=[embed_dim], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    # External kernel
    rope_func = FuncOp(
        "rope", ([l1RowTy, l1RowTy, l1RowTy, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    # Affine map: offset within column's chunk
    # ty_row_offset = (_ty * rows_per_tile + local_row) * embed_dim
    intra_col_map = AffineMap.get(
        0,
        2,  # s0=local_row, s1=_ty
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),  # local_row
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(1),  # _ty
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(embed_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3DataTy, l3DataTy, l3DataTy)
    def rope_lut(arg0, arg1, arg2):

        @launch(operands=[arg0, arg1, arg2])
        def launch_body(l_in, l_lut, l_out):

            @segment(name="rope_seg", operands=[l_in, l_lut, l_out])
            def segment_body(s_in, s_lut, s_out):
                l2_in = AllocOp(l2DataTy, [], [])
                l2_lut = AllocOp(l2DataTy, [], [])
                l2_out = AllocOp(l2DataTy, [], [])

                # Outer loop: iterate over batches
                for batch_offset in range_(0, total, batch_size_elems):
                    # DDR → L2 (one batch, no _tx/_ty dependency)
                    dma_memcpy_nd(
                        l2_in,
                        s_in,
                        src_offsets=[batch_offset],
                        src_sizes=[batch_size_elems],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l2_lut,
                        s_lut,
                        src_offsets=[batch_offset],
                        src_sizes=[batch_size_elems],
                        src_strides=[1],
                    )

                    @herd(
                        name="herd_0",
                        sizes=[herd_x, herd_y],
                        operands=[l2_in, l2_lut, l2_out],
                    )
                    def herd_body(_tx, _ty, _sx, _sy, h_in, h_lut, h_out):
                        l1_in = AllocOp(l1RowTy, [], [])
                        l1_lut = AllocOp(l1RowTy, [], [])
                        l1_out = AllocOp(l1RowTy, [], [])
                        dim_i32 = ConstantOp(T.i32(), embed_dim)

                        for local_row in range_(rows_per_tile):
                            # Offset within this column's L2 chunk
                            intra_off = affine_apply(intra_col_map, [local_row, _ty])

                            # L2 → L1: _tx selects MemTile (dim 0), intra_off within column
                            dma_memcpy_nd(
                                l1_in,
                                h_in,
                                src_offsets=[_tx, intra_off],
                                src_sizes=[1, embed_dim],
                                src_strides=[col_chunk, 1],
                            )
                            dma_memcpy_nd(
                                l1_lut,
                                h_lut,
                                src_offsets=[_tx, intra_off],
                                src_sizes=[1, embed_dim],
                                src_strides=[col_chunk, 1],
                            )

                            CallOp(rope_func, [l1_in, l1_lut, l1_out, dim_i32])

                            # L1 → L2
                            dma_memcpy_nd(
                                h_out,
                                l1_out,
                                dst_offsets=[_tx, intra_off],
                                dst_sizes=[1, embed_dim],
                                dst_strides=[col_chunk, 1],
                            )
                            yield_([])

                        DeallocOp(l1_in)
                        DeallocOp(l1_lut)
                        DeallocOp(l1_out)

                    herd_body.attributes["link_with"] = StringAttr.get("rope.o")

                    # L2 → DDR (write back batch)
                    dma_memcpy_nd(
                        s_out,
                        l2_out,
                        dst_offsets=[batch_offset],
                        dst_sizes=[batch_size_elems],
                        dst_strides=[1],
                    )
                    yield_([])

                DeallocOp(l2_in)
                DeallocOp(l2_lut)
                DeallocOp(l2_out)


from rope_lut import rope_reference, generate_lut

if __name__ == "__main__":
    THETA = 10000.0

    parser = argparse.ArgumentParser(
        description="RoPE LUT with tiled L2 staging (2D herd)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--herd-x", type=int, default=8)
    parser.add_argument("--herd-y", type=int, default=4)
    parser.add_argument("--rows-per-batch", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        choices=["xclbin", "elf"],
        default="xclbin",
    )
    args = parser.parse_args()

    seq_len = args.seq_len
    embed_dim = args.embed_dim
    herd_x = args.herd_x
    herd_y = args.herd_y
    total_tiles = herd_x * herd_y

    print(
        f"RoPE LUT (tiled L2): seq_len={seq_len}, embed_dim={embed_dim}, "
        f"herd=[{herd_x},{herd_y}] ({total_tiles} tiles)"
    )

    mlir_module = build_module(
        seq_len,
        embed_dim,
        bfloat16,
        herd_x=herd_x,
        herd_y=herd_y,
        rows_per_batch=args.rows_per_batch,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(0)
        input_data = np.random.uniform(-4.0, 4.0, (seq_len, embed_dim)).astype(bfloat16)
        lut = generate_lut(seq_len, embed_dim, bfloat16, THETA)
        y_expected = rope_reference(input_data, lut, embed_dim)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_data.flatten(), lut.flatten()],
                expected_outputs=[y_expected.flatten()],
                rtol=5e-2,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(mlir_module)
        backend.unload()
