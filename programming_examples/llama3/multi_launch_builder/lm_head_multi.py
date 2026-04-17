#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LM Head multi-launch — 8-partition GEMM in one ELF.

Partitions the large vocab projection (2048, 2048, 128256) into 8 GEMMs
of (2048, 2048, 16384) each, stitched as 8 air.launch ops in one ELF.
Last partition padded (128256 = 7*16384 + 13568, padded to 16384).

17 func args: 1 shared input + 8 weights + 8 outputs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _llm_shared.kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _rename_all,
    _fix_launch_func_args,
)


def build_lm_head_module(
    seq_len=2048,
    emb_dim=2048,
    n_partitions=8,
    n_part=16384,
    tile_m=64,
    tile_k_l2=64,
    tile_k_l1=32,
    tile_n=128,
    herd_m=8,
    herd_n=4,
):
    """Build multi-launch LM Head: n_partitions air.launch ops in one func.

    Returns:
        Module with func @lm_head and (1 + 2*n_partitions) memref args:
            %arg0: x (seq_len, emb_dim) — shared input
            %arg(1+2*p): weight_p (emb_dim, n_part)
            %arg(2+2*p): output_p (seq_len, n_part)
    """
    from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module

    print(f"  Building {n_partitions}-partition LM Head GEMM (N_part={n_part})...")
    gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            n_part,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    bodies, maps_all = [], []
    for p in range(n_partitions):
        prefix = f"p{p}"
        body = _extract_between_func_and_return(gemm_ir)
        maps = _extract_affine_maps(gemm_ir)
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, {0: 0, 1: 1 + 2 * p, 2: 2 + 2 * p})
        bodies.append(body)
        maps_all.extend(maps)

    arg_lines = [f"    %arg0: memref<{seq_len}x{emb_dim}xbf16>"]
    for p in range(n_partitions):
        arg_lines.append(f"    %arg{1+2*p}: memref<{emb_dim}x{n_part}xbf16>")
        arg_lines.append(f"    %arg{2+2*p}: memref<{seq_len}x{n_part}xbf16>")

    combined = "\n".join(maps_all) + f"""
module {{
  func.func @lm_head(
{(',' + chr(10)).join(arg_lines)}
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(
            f"  Module: {len(combined.splitlines())} lines, {1+2*n_partitions} args, {n_partitions} launches, parsed OK"
        )
        return module
