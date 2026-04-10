#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN Decode Full — 6-launch multi-launch ELF.

RMSNorm + Gate GEMV + Up GEMV + SiLU×mul + Down GEMV + Add in one ELF.
Same pattern as prefill's ffn_full_multi.py but with GEMV instead of GEMM.

func @ffn_decode_full(
    %arg0: memref<2048xbf16>,         # res1 (input, 1D)
    %arg1: memref<2048xbf16>,         # ffn_norm_weight
    %arg2: memref<2048xbf16>,         # normed2 (RMSNorm output, 1D)
    %arg3: memref<8192x2048xbf16>,    # w_gate (transposed)
    %arg4: memref<8192xbf16>,         # gate_out
    %arg5: memref<8192x2048xbf16>,    # w_up (transposed)
    %arg6: memref<8192xbf16>,         # up_out
    %arg7: memref<8192xbf16>,         # swiglu_out
    %arg8: memref<2048x8192xbf16>,    # w_down (transposed)
    %arg9: memref<2048xbf16>,         # down_out
    %arg10: memref<2048xbf16>         # output (res1 + down)
)

Note: RMSNorm outputs (1, 2048) 2D but GEMV expects (2048,) 1D.
We handle this by having the combined func use 1D for the normed buffer,
and wrapping RMSNorm to output 1D via collapse_shape inside its launch.
Actually, for simplicity, RMSNorm uses M=1 N=2048 which outputs memref<1x2048>.
We use a separate 1D arg2 for GEMV and let the host flatten between them.
Wait — in multi-launch, there's no "host flatten" — all launches run in one
xrt.run(). So we need the RMSNorm launch to write to the same buffer that
GEMV reads. The shape mismatch (2D vs 1D) requires collapse_shape.

Simpler approach: skip RMSNorm in this merge (keep it separate).
Merge only: Gate + Up + SiLU×mul + Down + Add (5 launches).
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
    _extract_private_funcs,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)
from llama3.multi_launch_builder.rms_qkv_gemv_multi import _rename_all_gemv


def build_ffn_decode_full_module(
    emb_dim=2048,
    hidden_dim=8192,
    gate_tile_m=8,
    gate_m_input=4,
    down_tile_m=2,
    down_m_input=1,
    herd_m=8,
):
    """Build 5-launch FFN decode: Gate + Up + SiLU×mul + Down + Add.

    RMSNorm stays separate (shape mismatch: 2D vs 1D).

    Args layout:
        arg0: normed2 (emb_dim,)         input vector (shared by gate+up)
        arg1: w_gate (hidden_dim, emb_dim) transposed weight
        arg2: gate_out (hidden_dim,)
        arg3: w_up (hidden_dim, emb_dim)
        arg4: up_out (hidden_dim,)
        arg5: swiglu_out (hidden_dim,)    SiLU×mul result
        arg6: w_down (emb_dim, hidden_dim) transposed weight
        arg7: down_out (emb_dim,)
        arg8: res1 (emb_dim,)            residual input
        arg9: output (emb_dim,)          res1 + down
    """
    from matvec import build_module as build_gemv
    from eltwise_add.eltwise_add import build_module as build_add
    import importlib.util

    _silu_path = os.path.join(
        os.path.dirname(__file__), "..", "ffn_swiglu", "silu_and_mul.py"
    )
    _spec = importlib.util.spec_from_file_location("silu_and_mul", _silu_path)
    _silu_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_silu_mod)

    # 1. Gate GEMV
    print("  [1/5] Gate GEMV...")
    gate_ir = str(
        build_gemv(
            hidden_dim, emb_dim, gate_tile_m, gate_m_input, herd_m, bfloat16, bfloat16
        )
    )

    # 2. Up GEMV
    print("  [2/5] Up GEMV...")
    up_ir = str(
        build_gemv(
            hidden_dim, emb_dim, gate_tile_m, gate_m_input, herd_m, bfloat16, bfloat16
        )
    )

    # 3. SiLU×mul (needs launch+segment wrapper)
    print("  [3/5] SiLU×mul...")
    silu_ir = _wrap_ir_in_launch(
        str(
            _silu_mod.build_module(
                hidden_dim, hidden_dim // 8, bfloat16, herd_x=8, herd_y=1
            )
        )
    )

    # 4. Down GEMV
    print("  [4/5] Down GEMV...")
    down_ir = str(
        build_gemv(
            emb_dim, hidden_dim, down_tile_m, down_m_input, herd_m, bfloat16, bfloat16
        )
    )

    # 5. Add (needs launch+segment wrapper)
    print("  [5/5] Add...")
    add_ir = _wrap_ir_in_launch(
        str(
            build_add(
                emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1
            )
        )
    )

    # Stitch all 5 launches
    # Gate: A=w_gate(arg1), B=normed2(arg0), C=gate_out(arg2)
    # Up:   A=w_up(arg3), B=normed2(arg0), C=up_out(arg4)
    # SiLU: gate(arg2), up(arg4), swiglu(arg5)  — 3 args mapped to {0->2, 1->4, 2->5}
    # Down: A=w_down(arg6), B=swiglu(arg5), C=down_out(arg7)
    # Add:  a=res1(arg8), b=down(arg7), c=output(arg9) — {0->8, 1->7, 2->9}

    bodies, maps_all = [], []

    # Gate GEMV
    body = _extract_between_func_and_return(gate_ir)
    maps = _extract_affine_maps(gate_ir)
    body = _rename_all_gemv(body, "g")
    maps = [_rename_all_gemv(m, "g") for m in maps]
    body = _fix_launch_func_args(body, "g", {0: 1, 1: 0, 2: 2})
    bodies.append(body)
    maps_all.extend(maps)

    # Up GEMV
    body = _extract_between_func_and_return(up_ir)
    maps = _extract_affine_maps(up_ir)
    body = _rename_all_gemv(body, "u")
    maps = [_rename_all_gemv(m, "u") for m in maps]
    body = _fix_launch_func_args(body, "u", {0: 3, 1: 0, 2: 4})
    bodies.append(body)
    maps_all.extend(maps)

    # SiLU×mul — use _rename_all (not _rename_all_gemv) since it has different extern funcs
    from llama3.multi_launch_builder.ffn_full_multi import (
        _rename_all as _rename_all_full,
    )

    body = _extract_between_func_and_return(silu_ir)
    maps = _extract_affine_maps(silu_ir)
    body = _rename_all_full(body, "s")
    maps = [_rename_all_full(m, "s") for m in maps]
    body = _fix_launch_func_args(body, "s", {0: 2, 1: 4, 2: 5})
    bodies.append(body)
    maps_all.extend(maps)

    # Down GEMV
    body = _extract_between_func_and_return(down_ir)
    maps = _extract_affine_maps(down_ir)
    body = _rename_all_gemv(body, "d")
    maps = [_rename_all_gemv(m, "d") for m in maps]
    body = _fix_launch_func_args(body, "d", {0: 6, 1: 5, 2: 7})
    bodies.append(body)
    maps_all.extend(maps)

    # Add
    body = _extract_between_func_and_return(add_ir)
    maps = _extract_affine_maps(add_ir)
    body = _rename_all_gemv(body, "a")
    maps = [_rename_all_gemv(m, "a") for m in maps]
    body = _fix_launch_func_args(body, "a", {0: 8, 1: 7, 2: 9})
    bodies.append(body)
    maps_all.extend(maps)

    # Collect private func declarations from all sub-kernels
    privates = set()
    for ir in [gate_ir, silu_ir]:
        privates.update(l.strip() for l in ir.split("\n") if "func.func private" in l)

    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p for p in privates)}
  func.func @ffn_decode_full(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg2: memref<{hidden_dim}xbf16>,
    %arg3: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg4: memref<{hidden_dim}xbf16>,
    %arg5: memref<{hidden_dim}xbf16>,
    %arg6: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg7: memref<{emb_dim}xbf16>,
    %arg8: memref<{emb_dim}xbf16>,
    %arg9: memref<{emb_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, 10 args, 5 launches")
        return module
