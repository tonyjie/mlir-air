#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Half-split RoPE — standalone correctness + profile harness.

Wraps the `@rope` external C++ kernel (rope_halfsplit.cc) in a self-contained
AIR module: 2D input (outer_rows, outer_cols) → flat → herd of `herd_x`
tiles, each tile DMA's one head_dim-wide row in/out and calls @rope.

Default shape mirrors llama3-1B Q-RoPE production:
    outer_rows = seq_len   = 2048
    outer_cols = emb_dim   = 2048   (= n_heads * head_dim = 32 * 64)
    head_dim   = 64
    herd_x     = 8

For K-RoPE (kv_dim = 512), pass --outer-cols 512.

Usage:
    make run                 # compile + correctness (cosine ≥ 0.99)
    make profile             # compile + run + per-iteration timing
    make print               # print MLIR module
    python3 run.py --outer-cols 512   # K-RoPE shape
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

# sys.path: programming_examples/ + _llm_shared/kernel_builder/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    collapse_shape as memref_collapse_shape,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# Module builder — copied from llama3/multi_launch_builder/rms_gemms_rope_multi.py
# (kept self-contained here so this harness has no llama3 dependency)
# ---------------------------------------------------------------------------


@module_builder
def build_module(outer_rows, outer_cols, embed_dim, np_dtype, herd_x):
    """Build a RoPE launch with 2D in/out args.

    Args:
        outer_rows: 2D func arg rows (e.g. seq_len=2048)
        outer_cols: 2D func arg cols (e.g. emb_dim=2048 or kv_dim=512)
        embed_dim:  RoPE column width per row (head_dim=64)
        herd_x:     Number of tiles for row-parallel
    """
    xrt_dtype = type_mapper(np_dtype)
    total = outer_rows * outer_cols
    rope_rows = total // embed_dim
    herd_y = 1
    total_tiles = herd_x * herd_y

    assert embed_dim % 16 == 0, "embed_dim must be divisible by 16 (AIE2P bf16 vector)"
    assert total % embed_dim == 0
    assert (
        rope_rows % total_tiles == 0
    ), f"rope_rows={rope_rows} must be divisible by total_tiles={total_tiles}"

    l3_2d_ty = MemRefType.get([outer_rows, outer_cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([total], xrt_dtype)

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get(
        shape=[embed_dim], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    rope_func = FuncOp(
        "rope", ([l1RowTy, l1RowTy, l1RowTy, T.i32()], []), visibility="private"
    )
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    rows_per_tile = rope_rows // total_tiles

    # Affine map: row_offset = (local_row + tile_id * rows_per_tile) * embed_dim
    #   tile_id = _tx * herd_y + _ty
    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),  # local_row
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),  # _tx
                                AffineConstantExpr.get(herd_y),
                            ),
                            AffineSymbolExpr.get(2),  # _ty
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(embed_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_2d_ty, l3_1d_ty, l3_2d_ty)
    def rope_halfsplit(arg0_2d, arg1_lut, arg2_2d):
        @launch(operands=[arg0_2d, arg1_lut, arg2_2d])
        def rope_launch(l_in_2d, l_lut, l_out_2d):
            in_flat = memref_collapse_shape(l3_1d_ty, l_in_2d, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out_2d, [[0, 1]])

            @segment(name="rope_seg", operands=[in_flat, l_lut, out_flat])
            def rope_seg(s_in, s_lut, s_out):
                @herd(
                    name="rope_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_lut, s_out],
                )
                def rope_body(_tx, _ty, _sx, _sy, h_in, h_lut, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_lut = AllocOp(l1RowTy, [], [])
                    l1_out = AllocOp(l1RowTy, [], [])

                    dim_i32 = ConstantOp(T.i32(), embed_dim)

                    for local_row in for_(rows_per_tile):
                        row_offset = affine_apply(row_offset_map, [local_row, _tx, _ty])

                        dma_memcpy_nd(
                            l1_in,
                            h_in,
                            src_offsets=[row_offset],
                            src_sizes=[embed_dim],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_lut,
                            h_lut,
                            src_offsets=[row_offset],
                            src_sizes=[embed_dim],
                            src_strides=[1],
                        )

                        CallOp(rope_func, [l1_in, l1_lut, l1_out, dim_i32])

                        dma_memcpy_nd(
                            h_out,
                            l1_out,
                            dst_offsets=[row_offset],
                            dst_sizes=[embed_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out)

                rope_body.attributes["link_with"] = StringAttr.get("rope.o")


# ---------------------------------------------------------------------------
# CPU reference (numpy) — half-split RoPE matching the C++ kernel
# ---------------------------------------------------------------------------


def rope_halfsplit_reference(input_2d, lut_1d, head_dim):
    """Per-row half-split RoPE in numpy F32, casts back to bfloat16.

    Mirrors the C++ kernel exactly:
        out[i]        = x[i] * cos[i] - x[i + half] * sin[i]
        out[i + half] = x[i] * sin[i] + x[i + half] * cos[i]
    where lut layout per row is [cos[0..half-1], sin[0..half-1]].

    Args:
        input_2d:  (outer_rows, outer_cols) bf16
        lut_1d:    (outer_rows * outer_cols,) bf16, conceptually
                   reshapable to (rope_rows, head_dim) where each
                   head_dim-row is [cos, sin] concatenated.
        head_dim:  RoPE inner dimension.

    Returns:
        (outer_rows, outer_cols) bf16
    """
    outer_rows, outer_cols = input_2d.shape
    total = outer_rows * outer_cols
    assert total % head_dim == 0
    rope_rows = total // head_dim
    half = head_dim // 2

    x = input_2d.astype(np.float32).reshape(rope_rows, head_dim)
    lut = lut_1d.astype(np.float32).reshape(rope_rows, head_dim)

    cos = lut[:, :half]
    sin = lut[:, half:]

    x1 = x[:, :half]
    x2 = x[:, half:]

    out = np.empty_like(x)
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x1 * sin + x2 * cos

    return out.reshape(outer_rows, outer_cols).astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Half-split RoPE standalone correctness + profile harness"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile kernel execution"
    )
    parser.add_argument(
        "--outer-rows", type=int, default=2048, help="seq_len (default: LLAMA seq_len)"
    )
    parser.add_argument(
        "--outer-cols",
        type=int,
        default=2048,
        help="Q: emb_dim (default 2048); K: kv_dim (e.g. 512)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="RoPE head_dim (default: LLAMA 64)"
    )
    parser.add_argument(
        "--herd-x", type=int, default=8, help="Number of row-parallel tiles"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Profiling timed iterations (after warmup)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Profiling warmup iterations (untimed)"
    )
    args = parser.parse_args()

    outer_rows = args.outer_rows
    outer_cols = args.outer_cols
    head_dim = args.head_dim
    herd_x = args.herd_x
    rope_rows = (outer_rows * outer_cols) // head_dim

    print(
        f"RoPE half-split: outer=({outer_rows},{outer_cols}), "
        f"head_dim={head_dim}, rope_rows={rope_rows}, herd_x={herd_x}"
    )

    module = build_module(outer_rows, outer_cols, head_dim, bfloat16, herd_x)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Build test inputs
    np.random.seed(42)
    x_2d = (np.random.randn(outer_rows, outer_cols) * 1.0).astype(bfloat16)

    # LUT layout: per-row [cos[0..half-1], sin[0..half-1]] in flat 1D form.
    # Use real position-based cos/sin so the test exercises typical magnitudes.
    half = head_dim // 2
    # Position-frequency table (rope_base ~ 10000 typical for testing; magnitude
    # invariant — we just need cos/sin ∈ [-1, 1])
    inv_freq = 1.0 / (10000 ** (np.arange(0, half) / half))
    # rope_rows worth of positions cycled — the kernel doesn't care about
    # which (head, position) combination this row is, only that LUT is per-row.
    positions = np.arange(rope_rows) % outer_rows  # cycle through seq positions
    angles = positions[:, None] * inv_freq[None, :]  # (rope_rows, half)
    lut_per_row = np.concatenate(
        [np.cos(angles), np.sin(angles)], axis=1
    )  # (rope_rows, head_dim)
    lut_1d = lut_per_row.reshape(-1).astype(bfloat16)
    out_2d = np.zeros((outer_rows, outer_cols), dtype=bfloat16)

    # CPU reference
    output_ref = rope_halfsplit_reference(x_2d, lut_1d, head_dim)

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=True,
            output_format="elf",
            instance_name="rope_halfsplit",
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [x_2d, lut_1d, out_2d]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        # Warmup (untimed)
        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for _ in range(args.warmup):
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()

        # Timed iterations
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

        # Read output and compute correlation
        output_bo = bos[-1]
        output_data = output_bo.read(sizes[-1], 0).view(np.int16).view(bfloat16)
        output_data = output_data.reshape(outer_rows, outer_cols).astype(np.float32)
        ref_flat = output_ref.astype(np.float32).flatten()
        corr = np.corrcoef(output_data.flatten(), ref_flat)[0, 1]
        cosine = np.dot(output_data.flatten(), ref_flat) / (
            np.linalg.norm(output_data.flatten()) * np.linalg.norm(ref_flat) + 1e-12
        )

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel:           avg={np.mean(times_kernel):.3f}ms  "
            f"min={np.min(times_kernel):.3f}ms  max={np.max(times_kernel):.3f}ms"
        )
        print(
            f"  Total (w+r+rd):   avg={np.mean(times_total):.3f}ms  "
            f"min={np.min(times_total):.3f}ms  max={np.max(times_total):.3f}ms"
        )
        print(
            f"  Host overhead:    {np.mean(times_total) - np.mean(times_kernel):.3f}ms"
        )
        print(f"  Correlation:      {corr:.6f}")
        print(f"  Cosine sim:       {cosine:.6f}")
        # Expected: cosine ≥ 0.99 vs CPU F32 reference (BF16 quantization)
        status = "PASS" if cosine > 0.99 else "FAIL"
        print(f"\n  {status} (cosine={cosine:.6f})")

    else:
        # Correctness test via XRTRunner (cosine check internally)
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=True,
            output_format="elf",
            instance_name="rope_halfsplit",
        )
        exit(
            runner.run_test(
                module,
                inputs=[x_2d, lut_1d],
                expected_outputs=[output_ref],
                rtol=0.04,
                atol=0.05,
                min_correlation=0.99,
            )
        )
