#!/usr/bin/env python3
"""Standalone test for the Q4 GEMV kernel on NPU.

Concatenates packed Q4 + scales + mins into one buffer to minimize DMA channels.
Uses a [1,1] herd for simplicity.

Run from build_peano/:
  python3 ../test/test_mv_q4.py
"""

import os
import sys
import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from air.ir import *
from air.dialects.air import *
from air.dialects import arith, func as func_dialect
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from llama3.kernel_builder.quantize import pack_q4_for_npu, q4_dequant_reference

# Test dimensions
M = 4
K = 2048
BLOCK_SIZE = 32
N_BLOCKS_PER_ROW = K // BLOCK_SIZE  # 64
PACKED_BYTES = M * K // 2  # 4096
SCALES_BYTES = M * N_BLOCKS_PER_ROW * 2  # 512 (bf16 = 2 bytes)
MINS_BYTES = M * N_BLOCKS_PER_ROW * 2  # 512
# Concatenated weight buffer (all as uint8)
WEIGHT_BYTES = PACKED_BYTES + SCALES_BYTES + MINS_BYTES  # 5120


@module_builder
def build_q4_gemv_module():
    xrt_bf16 = type_mapper(bfloat16)
    xrt_i8 = IntegerType.get_unsigned(8)
    l1_mem = IntegerAttr.get(T.i32(), 2)

    # L3 types: weight_buf (concat), input vec, output vec
    weight_ty = MemRefType.get([WEIGHT_BYTES], xrt_i8)
    b_ty = MemRefType.get([K], xrt_bf16)
    c_ty = MemRefType.get([M], xrt_bf16)

    # L1 types
    l1_weight_ty = MemRefType.get([WEIGHT_BYTES], xrt_i8, memory_space=l1_mem)
    l1_b_ty = MemRefType.get([K], xrt_bf16, memory_space=l1_mem)
    l1_c_ty = MemRefType.get([M], xrt_bf16, memory_space=l1_mem)

    # Split L1 types for kernel call (views into the concat buffer)
    l1_packed_ty = MemRefType.get([PACKED_BYTES], xrt_i8, memory_space=l1_mem)
    l1_scales_ty = MemRefType.get([M * N_BLOCKS_PER_ROW], xrt_bf16, memory_space=l1_mem)
    l1_mins_ty = MemRefType.get([M * N_BLOCKS_PER_ROW], xrt_bf16, memory_space=l1_mem)

    # External kernel
    q4_func_ty = FunctionType.get(
        [
            T.i32(),
            T.i32(),
            T.i32(),
            l1_packed_ty,
            l1_scales_ty,
            l1_mins_ty,
            l1_b_ty,
            l1_c_ty,
        ],
        [],
    )
    q4_func = FuncOp("q4_matvec_bf16", q4_func_ty, visibility="private")
    q4_func.attributes["link_with"] = StringAttr.get("mv_q4.o")

    @FuncOp.from_py_func(weight_ty, b_ty, c_ty)
    def q4_gemv(l3_weight, l3_b, l3_c):
        @launch(operands=[l3_weight, l3_b, l3_c])
        def launch_body(l_weight, l_b, l_c):
            @segment(name="seg0", operands=[l_weight, l_b, l_c])
            def seg_body(s_weight, s_b, s_c):
                @herd(name="herd0", sizes=[1, 1], operands=[s_weight, s_b, s_c])
                def herd_body(_tx, _ty, _sx, _sy, h_weight, h_b, h_c):
                    l1_weight = AllocOp(l1_weight_ty, [], [])
                    l1_b = AllocOp(l1_b_ty, [], [])
                    l1_c = AllocOp(l1_c_ty, [], [])

                    # DMA: L3 → L1
                    dma_memcpy_nd(
                        l1_weight,
                        h_weight,
                        src_offsets=[],
                        src_sizes=[WEIGHT_BYTES],
                        src_strides=[1],
                    )
                    dma_memcpy_nd(
                        l1_b, h_b, src_offsets=[], src_sizes=[K], src_strides=[1]
                    )

                    # Reinterpret views into the concatenated weight buffer
                    from air.dialects.memref import reinterpret_cast

                    l1_packed = reinterpret_cast(
                        l1_packed_ty,
                        l1_weight,
                        [],
                        [PACKED_BYTES],
                        [1],
                        static_offsets=[0],
                        static_sizes=[PACKED_BYTES],
                        static_strides=[1],
                    )
                    l1_scales = reinterpret_cast(
                        l1_scales_ty,
                        l1_weight,
                        [],
                        [M * N_BLOCKS_PER_ROW],
                        [1],
                        static_offsets=[PACKED_BYTES // 2],  # offset in bf16 units
                        static_sizes=[M * N_BLOCKS_PER_ROW],
                        static_strides=[1],
                    )
                    l1_mins = reinterpret_cast(
                        l1_mins_ty,
                        l1_weight,
                        [],
                        [M * N_BLOCKS_PER_ROW],
                        [1],
                        static_offsets=[PACKED_BYTES // 2 + M * N_BLOCKS_PER_ROW],
                        static_sizes=[M * N_BLOCKS_PER_ROW],
                        static_strides=[1],
                    )

                    m_val = arith.ConstantOp(T.i32(), M)
                    k_val = arith.ConstantOp(T.i32(), K)
                    zero = arith.ConstantOp(T.i32(), 0)
                    func_dialect.CallOp(
                        q4_func,
                        [m_val, k_val, zero, l1_packed, l1_scales, l1_mins, l1_b, l1_c],
                    )

                    dma_memcpy_nd(
                        h_c, l1_c, src_offsets=[], src_sizes=[M], src_strides=[1]
                    )

                    DeallocOp(l1_weight)
                    DeallocOp(l1_b)
                    DeallocOp(l1_c)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Q4 GEMV kernel test")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    args = parser.parse_args()

    print(f"Q4 GEMV Test: M={M}, K={K}, block_size={BLOCK_SIZE}")

    module = build_q4_gemv_module()

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Generate test data
    np.random.seed(42)
    w_bf16 = np.random.uniform(-1, 1, (M, K)).astype(bfloat16)
    x_bf16 = np.random.uniform(-1, 1, K).astype(bfloat16)

    # Pack to Q4
    packed, scales, mins = pack_q4_for_npu(w_bf16)

    # Concatenate into single buffer (as uint8)
    weight_buf = np.concatenate(
        [
            packed.flatten(),
            scales.view(np.uint8).flatten(),
            mins.view(np.uint8).flatten(),
        ]
    ).astype(np.uint8)
    print(f"  Weight buffer: {weight_buf.shape} ({weight_buf.nbytes} bytes)")

    # CPU reference
    w_dequant = q4_dequant_reference(packed, scales, mins, M, K, BLOCK_SIZE)
    y_ref = (w_dequant @ x_bf16.astype(np.float32)).astype(bfloat16)
    print(f"  CPU reference: {y_ref}")

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=True,
            output_format="elf",
            instance_name="q4_gemv_test",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    c_out = np.zeros(M, dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=True,
        output_format="elf",
        instance_name="q4_gemv_test",
    )
    exit_code = runner.run_test(
        module,
        inputs=[weight_buf, x_bf16, c_out],
        expected_outputs=[y_ref],
        rtol=0.05,
        atol=0.1,
    )
    sys.exit(exit_code)
