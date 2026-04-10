#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""O GEMV + Residual Add — 2-launch multi-launch ELF for decode.

func @o_gemv_add(
    %arg0: memref<2048x2048xbf16>,  # wo (transposed weight)
    %arg1: memref<2048xbf16>,        # attn_out (input vector)
    %arg2: memref<2048xbf16>,        # proj_out (GEMV output)
    %arg3: memref<2048xbf16>,        # x_residual (residual input)
    %arg4: memref<2048xbf16>,        # output (add result)
)
"""

import os
import sys
import re

from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "matrix_vector_multiplication", "bf16"
    ),
)

from llama3.multi_launch_builder.ffn_full_multi import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)
from llama3.multi_launch_builder.rms_qkv_gemv_multi import _rename_all_gemv


def build_o_gemv_add_module(emb_dim=2048, tile_m=8, m_input=4, herd_m=8):
    """Build multi-launch O GEMV + Add for decode."""
    from matvec import build_module as build_gemv
    from eltwise_add.eltwise_add import build_module as build_add

    print("  [1/2] O GEMV...")
    gemv_ir = str(
        build_gemv(emb_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print("  [2/2] Eltwise Add...")
    add_ir = _wrap_ir_in_launch(
        str(build_add(emb_dim, 1024, bfloat16, vector_size=16, herd_x=1, herd_y=1))
    )

    bodies, maps_all = [], []

    # O GEMV: {0->0, 1->1, 2->2} (wo, attn_out, proj_out)
    body = _extract_between_func_and_return(gemv_ir)
    maps = _extract_affine_maps(gemv_ir)
    body = _rename_all_gemv(body, "o")
    maps = [_rename_all_gemv(m, "o") for m in maps]
    body = _fix_launch_func_args(body, "o", {0: 0, 1: 1, 2: 2})
    bodies.append(body)
    maps_all.extend(maps)

    # Add: {0->3, 1->2, 2->4} (x_residual, proj_out, output)
    body = _extract_between_func_and_return(add_ir)
    maps = _extract_affine_maps(add_ir)
    body = _rename_all_gemv(body, "a")
    maps = [_rename_all_gemv(m, "a") for m in maps]
    body = _fix_launch_func_args(body, "a", {0: 3, 1: 2, 2: 4})
    bodies.append(body)
    maps_all.extend(maps)

    combined = "\n".join(maps_all) + f"""
module {{
  func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<{m_input}x{emb_dim}xbf16, 2 : i32>, memref<{emb_dim}xbf16, 2 : i32>, memref<{tile_m}xbf16, 2 : i32>) attributes {{link_with = "mv.o", llvm.emit_c_interface}}
  func.func private @linalg_fill_bf16(bf16, memref<{tile_m}xbf16, 2 : i32>) attributes {{link_with = "mv.o", llvm.emit_c_interface}}
  func.func @o_gemv_add(
    %arg0: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, 5 args, 2 launches")
        return module
