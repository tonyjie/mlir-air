#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O Projection GEMM + Residual Add — 2-launch multi-launch ELF.

Builds a single AIR function with 2 sequential air.launch operations:
  1. O GEMM       [8,4]   attn_out x wo -> proj_out
  2. Eltwise Add  [8,1]   proj_out + x_residual -> output

5 func args (2 launches). The eltwise add inputs (proj_out, x_residual)
are 2D memrefs with memref.collapse_shape to 1D inside the launch body.

Usage:
    python3 o_proj_add_multi.py -p           # print combined MLIR
    python3 o_proj_add_multi.py              # compile + run + validate
    python3 o_proj_add_multi.py --profile    # compile + run + profile
"""

import argparse
import os
import re
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

# ---------------------------------------------------------------------------
# MLIR text stitching utilities (shared with ffn_full_multi.py)
# ---------------------------------------------------------------------------

from ffn_full_multi import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _rename_all,
    _fix_launch_func_args,
)

# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_o_proj_add_module(
    seq_len=2048,
    emb_dim=2048,
    # O GEMM tile config
    tile_m=64,
    tile_k_l2=256,
    tile_k_l1=32,
    tile_n=64,
    herd_m=8,
    herd_n=4,
    # Add config
    add_tile_n=None,
    add_herd_x=8,
    add_herd_y=1,
    add_vector_size=16,
    print_kernels=False,
):
    """Build multi-launch O Proj + Add: 2 air.launch ops in one func.

    Args:
        seq_len: Sequence length (M dimension for GEMM).
        emb_dim: Embedding dimension (K and N for O GEMM).
        print_kernels: If True, print each sub-kernel's MLIR before stitching.
    """
    from llama3.llama3_prefill import _build_gemm_module

    if add_tile_n is None:
        add_tile_n = emb_dim

    n_total = seq_len * emb_dim

    # Build O GEMM
    print("  [1/2] O GEMM...")
    gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    # Build 2D eltwise add with collapse_shape inside launch
    print("  [2/2] Eltwise Add...")

    @module_builder
    def _build_add_2d():
        """Eltwise add accepting 2D memrefs, collapsing to 1D inside launch."""
        from air.dialects.memref import collapse_shape as memref_collapse_shape

        xrt_dtype = type_mapper(bfloat16)
        n = n_total
        l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        l3_1d_ty = MemRefType.get([n], xrt_dtype)

        total_tiles = add_herd_x * add_herd_y
        chunk_size = n // total_tiles
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([add_tile_n], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([add_vector_size], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
        def eltwise_add_2d(arg0_2d, arg1_2d, arg2_1d):
            @launch(operands=[arg0_2d, arg1_2d, arg2_1d])
            def add_launch(l_a, l_b, l_out):
                a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
                b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

                @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
                def add_seg(s_a, s_b, s_out):
                    offset_map = AffineMap.get(
                        0,
                        3,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineExpr.get_add(
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(1),
                                            AffineConstantExpr.get(add_herd_y),
                                        ),
                                        AffineSymbolExpr.get(2),
                                    ),
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )

                    @herd(
                        name="add_herd",
                        sizes=[add_herd_x, add_herd_y],
                        operands=[s_a, s_b, s_out],
                    )
                    def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                        l1_a = AllocOp(l1_ty, [], [])
                        l1_b = AllocOp(l1_ty, [], [])
                        l1_out = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                        for loop_iv in range_(0, chunk_size, add_tile_n):
                            offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                            dma_memcpy_nd(
                                l1_a,
                                h_a,
                                src_offsets=[offset],
                                src_sizes=[add_tile_n],
                                src_strides=[1],
                            )
                            dma_memcpy_nd(
                                l1_b,
                                h_b,
                                src_offsets=[offset],
                                src_sizes=[add_tile_n],
                                src_strides=[1],
                            )
                            for j in range_(0, add_tile_n, add_vector_size):
                                sub_a = subview(
                                    l1_a.result, [j], [add_vector_size], [1]
                                )
                                sub_b = subview(
                                    l1_b.result, [j], [add_vector_size], [1]
                                )
                                sub_out = subview(
                                    l1_out.result, [j], [add_vector_size], [1]
                                )
                                v_a = transfer_read(
                                    vec_ty, sub_a, [c0], identity_map, cst0, [True]
                                )
                                v_b = transfer_read(
                                    vec_ty, sub_b, [c0], identity_map, cst0, [True]
                                )
                                v_sum = arith.addf(v_a, v_b)
                                transfer_write(
                                    None, v_sum, sub_out, [c0], identity_map, [True]
                                )
                                yield_([])
                            dma_memcpy_nd(
                                h_out,
                                l1_out,
                                dst_offsets=[offset],
                                dst_sizes=[add_tile_n],
                                dst_strides=[1],
                            )
                            yield_([])
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_out)

    add_ir = str(_build_add_2d())

    if print_kernels:
        for name, ir in [("O GEMM", gemm_ir), ("Eltwise Add", add_ir)]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Stitch: O GEMM (prefix "o") + Add (prefix "a")
    bodies, maps_all = [], []

    # O GEMM: arg0=attn_out, arg1=wo, arg2=proj_out
    gemm_body = _extract_between_func_and_return(gemm_ir)
    gemm_maps = _extract_affine_maps(gemm_ir)
    gemm_body = _rename_all(gemm_body, "o")
    gemm_maps = [_rename_all(m, "o") for m in gemm_maps]
    gemm_body = _fix_launch_func_args(gemm_body, "o", {0: 0, 1: 1, 2: 2})
    bodies.append(gemm_body)
    maps_all.extend(gemm_maps)

    # Eltwise Add: arg0=proj_out(2D), arg1=x_residual(2D), arg2=output(1D)
    add_body = _extract_between_func_and_return(add_ir)
    add_maps = _extract_affine_maps(add_ir)
    add_body = _rename_all(add_body, "a")
    add_maps = [_rename_all(m, "a") for m in add_maps]
    add_body = _fix_launch_func_args(add_body, "a", {0: 2, 1: 3, 2: 4})
    bodies.append(add_body)
    maps_all.extend(add_maps)

    # Assemble the combined module (5 func args, 2 launches)
    combined = "\n".join(maps_all) + f"""
module {{
  func.func @o_proj_add(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{seq_len}x{emb_dim}xbf16>,
    %arg4: memref<{n_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def o_proj_add_reference(attn_out, wo, x_residual):
    """CPU F32 reference: O GEMM + Residual Add.

    Args:
        attn_out: (seq_len, emb_dim) attention output
        wo: (emb_dim, emb_dim) O projection weight
        x_residual: (seq_len, emb_dim) residual state

    Returns:
        output: (seq_len * emb_dim,) = x_residual + attn_out @ wo (flat)
    """
    proj = attn_out.astype(np.float32) @ wo.astype(np.float32)
    output = x_residual.astype(np.float32) + proj
    return output.astype(bfloat16).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O GEMM + Add multi-launch test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    seq_len, emb_dim = args.seq_len, args.emb_dim
    n_total = seq_len * emb_dim
    print(f"O Proj + Add Multi-Launch: seq_len={seq_len}, emb_dim={emb_dim}")

    module = build_o_proj_add_module(
        seq_len,
        emb_dim,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data
    np.random.seed(42)
    attn_out = (np.random.randn(seq_len, emb_dim) * 0.5).astype(bfloat16)
    wo = (np.random.randn(emb_dim, emb_dim) * 0.02).astype(bfloat16)
    proj_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    x_residual = (np.random.randn(seq_len, emb_dim) * 1.0).astype(bfloat16)
    output_buf = np.zeros(n_total, dtype=bfloat16)

    # CPU reference
    output_ref = o_proj_add_reference(attn_out, wo, x_residual)

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="o_proj_add",
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [attn_out, wo, proj_out, x_residual, output_buf]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        # Warmup
        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
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
            bos[4].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            t1 = time.perf_counter()
            times_kernel.append((tk1 - tk0) * 1000)
            times_total.append((t1 - t0) * 1000)

        # Check correctness
        output_data = bos[4].read(sizes[4], 0).view(np.int16).view(bfloat16)
        output_data = output_data.astype(np.float32)
        ref_flat = output_ref.astype(np.float32)
        corr = np.corrcoef(output_data, ref_flat)[0, 1]

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel (2 launches): avg={np.mean(times_kernel):.1f}ms  "
            f"min={np.min(times_kernel):.1f}ms  max={np.max(times_kernel):.1f}ms"
        )
        print(
            f"  Total (write+run+read): avg={np.mean(times_total):.1f}ms  "
            f"min={np.min(times_total):.1f}ms  max={np.max(times_total):.1f}ms"
        )
        print(f"  Host overhead: {np.mean(times_total) - np.mean(times_kernel):.1f}ms")
        print(f"  Correlation: {corr:.6f}")
        status = "PASS" if corr > 0.99 else "FAIL"
        print(f"\n  {status} (corr={corr:.6f})")

    else:
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="o_proj_add",
        )
        exit(
            runner.run_test(
                module,
                inputs=[attn_out, wo, proj_out, x_residual],
                expected_outputs=[output_ref],
                rtol=0.04,
                atol=4.0,
                min_correlation=0.99,
            )
        )
