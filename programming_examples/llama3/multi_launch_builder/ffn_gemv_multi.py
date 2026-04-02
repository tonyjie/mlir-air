#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN Decode — gate + up GEMVs merged into 2-launch ELF.

Both share the same input vector (normed2), saving one dispatch.
Down GEMV stays separate (needs CPU SiLU×mul between gate/up and down).

func @gate_up_gemv(
    %arg0: memref<8192x2048xbf16>,   # w_gate (transposed)
    %arg1: memref<2048xbf16>,         # normed2 (input vector, shared)
    %arg2: memref<8192xbf16>,         # gate_out
    %arg3: memref<8192x2048xbf16>,   # w_up (transposed)
    %arg4: memref<8192xbf16>,         # up_out
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
)
from llama3.multi_launch_builder.rms_qkv_gemv_multi import _rename_all_gemv


def build_gate_up_gemv_module(
    emb_dim=2048,
    hidden_dim=8192,
    tile_m=8,
    m_input=4,
    herd_m=8,
):
    """Build 2-launch gate + up GEMVs in one ELF.

    Both share normed2 input vector (arg1).
    """
    from matvec import build_module as build_gemv

    print("  [1/2] Gate GEMV...")
    gate_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print("  [2/2] Up GEMV...")
    up_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (gate_ir, "g", {0: 0, 1: 1, 2: 2}),
        (up_ir, "u", {0: 3, 1: 1, 2: 4}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_gemv(body, prefix)
        maps = [_rename_all_gemv(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    privates = [l for l in gate_ir.split("\n") if "func.func private" in l]

    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p.strip() for p in privates)}
  func.func @gate_up_gemv(
    %arg0: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{hidden_dim}xbf16>,
    %arg3: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg4: memref<{hidden_dim}xbf16>
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
