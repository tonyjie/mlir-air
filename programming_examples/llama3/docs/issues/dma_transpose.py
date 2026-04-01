#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DMA-based transpose on NPU — no compute kernel, pure data movement.

Transposes a 2D matrix (M, K) → (K, M) using strided DMA writes.
Based on the pattern from data_transfer_transpose/dma/transpose.py.

For the LLAMA pipeline, this transposes:
  (seq_len, n_heads * head_dim) → (n_heads * seq_len, head_dim)
which is equivalent to:
  reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2).reshape(n_heads*seq_len, head_dim)

Tiled version: processes tile_m rows at a time to fit in L1.

Usage:
    python3 dma_transpose.py              # test Q transpose
    python3 dma_transpose.py --profile    # profile
    python3 dma_transpose.py -p           # print MLIR
"""

import argparse
import os
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
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


def build_transpose_module(m, k, tile_m=64, herd_x=1, herd_y=1):
    """Build DMA-only transpose: (m, k) → (k, m).

    Uses the strided DMA write pattern from data_transfer_transpose.
    Tiles m dimension: each iteration loads tile_m rows into L1,
    writes back with transposed strides.

    L1 usage: tile_m * k * 2 bytes (e.g., 64*2048*2 = 256KB — need smaller tile or per-head tiling).
    """

    @module_builder
    def mod():
        xrt_dtype = type_mapper(bfloat16)
        memref_in = MemRefType.get([m, k], xrt_dtype)
        memref_out = MemRefType.get([k, m], xrt_dtype)

        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_ty = MemRefType.get([tile_m * k], xrt_dtype, memory_space=l1_space)

        @FuncOp.from_py_func(memref_in, memref_out)
        def transpose(src, dst):
            @launch(operands=[src, dst])
            def launch_body(l_src, l_dst):
                @segment(name="transpose_seg", operands=[l_src, l_dst])
                def seg_body(s_src, s_dst):
                    @herd(
                        name="transpose_herd",
                        sizes=[herd_x, herd_y],
                        operands=[s_src, s_dst],
                    )
                    def herd_body(tx, ty, sx, sy, h_src, h_dst):
                        l1_buf = AllocOp(l1_ty, [], [])

                        for row_start in range_(0, m, tile_m):
                            # Read tile_m contiguous rows from source
                            dma_memcpy_nd(
                                l1_buf,
                                h_src,
                                src_offsets=[row_start, 0],
                                src_sizes=[tile_m, k],
                                src_strides=[k, 1],
                            )

                            # Write back with transposed strides:
                            # L1 data is [row0_col0..col_k-1, row1_col0..col_k-1, ...]
                            # We want dst[col][row] = src[row][col]
                            # So for each col c, write tile_m values at dst[c, row_start:row_start+tile_m]
                            # Using 3D DMA: iterate over k columns, each writing tile_m elements
                            dma_memcpy_nd(
                                h_dst,
                                l1_buf,
                                src_sizes=[1, k, tile_m],
                                src_strides=[1, 1, k],
                                dst_offsets=[0, row_start],
                                dst_sizes=[k, tile_m],
                                dst_strides=[m, 1],
                            )
                            yield_([])

                        DeallocOp(l1_buf)

    return mod()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMA transpose test")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-m", type=int, default=64, help="M dimension")
    parser.add_argument("-k", type=int, default=32, help="K dimension")
    parser.add_argument("--tile-m", type=int, default=None, help="Tile M (auto)")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    m, k = args.m, args.k
    tile_m = args.tile_m or min(m, 32)  # default: fit in L1
    print(f"DMA Transpose: ({m}, {k}) → ({k}, {m}), tile_m={tile_m}")

    module = build_transpose_module(m, k, tile_m=tile_m)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    src = np.random.randint(0, 100, (m, k), dtype=np.int32).astype(bfloat16)
    ref = src.astype(np.float32).T.astype(bfloat16)

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="transpose",
        )
        artifact = backend.compile(module)
        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [src, np.zeros((k, m), dtype=bfloat16)]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16), 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()

        times = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        bos[1].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        npu = bos[1].read(sizes[1], 0).view(np.int16).view(bfloat16).astype(np.float32)
        ref_flat = ref.astype(np.float32).flatten()
        corr = np.corrcoef(npu, ref_flat)[0, 1]

        backend.unload()
        print(f"\nKernel: avg={np.mean(times):.2f}ms  min={np.min(times):.2f}ms")
        print(f"Correlation: {corr:.6f}")
        print(f"{'PASS' if corr > 0.999 else 'FAIL'}")

    else:
        runner = XRTRunner(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="transpose",
        )
        exit(
            runner.run_test(
                module,
                inputs=[src],
                expected_outputs=[ref],
                rtol=0,
                atol=0,
                min_correlation=0.999,
            )
        )
