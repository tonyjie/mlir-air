#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE 1D (decode-shape) — standalone correctness + profile harness.

Llama3-1B decode uses a different RoPE builder than prefill:
`_build_rope_1d(n_rows, embed_dim, herd_x)` in
`llama3/multi_launch_builder/rms_gemv_rope_multi.py`.

The 1D builder takes 1D `memref<total>` func args (matching GEMV layout
downstream) and processes `n_rows` rows of `embed_dim` elements each via
the same `@rope` external kernel as the 2D variant.

Default shape mirrors llama3-1B Q-RoPE decode:
    n_rows = 32  (= n_heads)
    embed_dim = 64  (= head_dim)
    herd_x = 1   (single-tile; production uses rope_herd_x=1)

For K-RoPE, pass --n-rows 8.

Tolerances:
    rtol = 4e-2, atol = 5e-2, min_correlation = 0.99
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

# ---------------------------------------------------------------------------
# Module builder — copied verbatim from llama3/multi_launch_builder/
#                   rms_gemv_rope_multi.py:_build_rope_1d
# ---------------------------------------------------------------------------


@module_builder
def build_module(n_rows, embed_dim, np_dtype=bfloat16, herd_x=1):
    """RoPE 1D launch: 1D func args, herd processes n_rows of embed_dim each."""
    xrt_dtype = type_mapper(np_dtype)
    total = n_rows * embed_dim
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert embed_dim % 16 == 0
    assert n_rows % total_tiles == 0

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

    rows_per_tile = n_rows // total_tiles

    row_offset_map = AffineMap.get(
        0,
        3,
        [
            AffineExpr.get_mul(
                AffineExpr.get_add(
                    AffineSymbolExpr.get(0),
                    AffineExpr.get_mul(
                        AffineExpr.get_add(
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1), AffineConstantExpr.get(herd_y)
                            ),
                            AffineSymbolExpr.get(2),
                        ),
                        AffineConstantExpr.get(rows_per_tile),
                    ),
                ),
                AffineConstantExpr.get(embed_dim),
            )
        ],
    )

    @FuncOp.from_py_func(l3_1d_ty, l3_1d_ty, l3_1d_ty)
    def rope_1d(arg0_in, arg1_lut, arg2_out):
        @launch(operands=[arg0_in, arg1_lut, arg2_out])
        def rope_launch(l_in, l_lut, l_out):
            @segment(name="rope_seg", operands=[l_in, l_lut, l_out])
            def rope_seg(s_in, s_lut, s_out):
                @herd(
                    name="rope_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_in, s_lut, s_out],
                )
                def rope_body(_tx, _ty, _sx, _sy, h_in, h_lut, h_out):
                    l1_in = AllocOp(l1RowTy, [], [])
                    l1_lut = AllocOp(l1RowTy, [], [])
                    l1_out_buf = AllocOp(l1RowTy, [], [])

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
                        CallOp(rope_func, [l1_in, l1_lut, l1_out_buf, dim_i32])
                        dma_memcpy_nd(
                            h_out,
                            l1_out_buf,
                            dst_offsets=[row_offset],
                            dst_sizes=[embed_dim],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out_buf)

                rope_body.attributes["link_with"] = StringAttr.get("rope.o")


def rope_halfsplit_reference(input_1d, lut_1d, n_rows, head_dim):
    """numpy F32 reference — same as rope_halfsplit/run.py but flat 1D."""
    half = head_dim // 2
    x = input_1d.astype(np.float32).reshape(n_rows, head_dim)
    lut = lut_1d.astype(np.float32).reshape(n_rows, head_dim)
    cos = lut[:, :half]
    sin = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x1 * sin + x2 * cos
    return out.reshape(-1).astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RoPE 1D (decode shape) correctness + profile"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--n-rows", type=int, default=32, help="32 = Q (n_heads), 8 = K (n_kv_heads)"
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--herd-x", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    n_rows, head_dim, herd_x = args.n_rows, args.head_dim, args.herd_x
    total = n_rows * head_dim
    print(
        f"RoPE 1D (decode): n_rows={n_rows}, head_dim={head_dim}, total={total}, herd_x={herd_x}"
    )

    module = build_module(n_rows, head_dim, bfloat16, herd_x)
    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    x_in = (np.random.randn(total) * 1.0).astype(bfloat16)
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (np.arange(0, half) / half))
    positions = np.arange(n_rows) % max(n_rows, 1)
    angles = positions[:, None] * inv_freq[None, :]
    lut_per_row = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    lut_1d = lut_per_row.reshape(-1).astype(bfloat16)
    out_buf = np.zeros((total,), dtype=bfloat16)

    output_ref = rope_halfsplit_reference(x_in, lut_1d, n_rows, head_dim)

    RTOL, ATOL, MIN_CORR = 4e-2, 5e-2, 0.99
    print(f"  Tolerances: rtol={RTOL}, atol={ATOL}, min_correlation={MIN_CORR}")

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=True,
            output_format="elf",
            instance_name="rope_1d",
        )
        artifact = backend.compile(module)
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [x_in, lut_1d, out_buf]
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

        out = bos[-1].read(sizes[-1], 0).view(np.int16).view(bfloat16).reshape(total)
        out_f, ref_f = out.astype(np.float32), output_ref.astype(np.float32)
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
            omit_while_true_loop=True,
            output_format="elf",
            instance_name="rope_1d",
        )
        exit(
            runner.run_test(
                module,
                inputs=[x_in, lut_1d],
                expected_outputs=[output_ref],
                rtol=RTOL,
                atol=ATOL,
                min_correlation=MIN_CORR,
            )
        )
