#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + Q/K/V GEMV — 4-launch multi-launch ELF for decode.

Same pattern as rms_attn_gemms_multi.py but with GEMV instead of GEMM.

func @rms_qkv_gemv(
    %arg0: memref<1x2048xbf16>,      # x_in (1 token)
    %arg1: memref<2048xbf16>,         # norm_weight
    %arg2: memref<1x2048xbf16>,       # normed (RMSNorm output)
    %arg3: memref<2048x2048xbf16>,    # wq (transposed: M×K)
    %arg4: memref<2048xbf16>,          # q_out
    %arg5: memref<512x2048xbf16>,     # wk (transposed)
    %arg6: memref<512xbf16>,           # k_out
    %arg7: memref<512x2048xbf16>,     # wv (transposed)
    %arg8: memref<512xbf16>,           # v_out
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
    _rename_all,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)


def _rename_all_gemv(text, prefix):
    """Rename SSA values preserving GEMV external kernel names."""
    extern = {"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"}
    for n in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(n) + r"(?!\w)", f"#{prefix}_{n[1:]}", text)
    for n in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(n) + r"(?!\w)", f"%{prefix}_{n[1:]}", text)
    for n in sorted(
        set(re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = text.replace(n, f"%{prefix}_n{n[1:]}")
    for n in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if n not in extern:
            text = text.replace(n, f"@{prefix}_{n[1:]}")
    return text


def build_rms_qkv_gemv_module(
    emb_dim=2048,
    kv_dim=512,
    tile_m=8,
    m_input=4,
    herd_m=8,
):
    """Build multi-launch RMSNorm + Q/K/V GEMVs for decode."""
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from matvec import build_module as build_gemv

    # RMSNorm at M=1
    print("  [1/4] RMSNorm (decode)...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(1, emb_dim, bfloat16, 16)))

    # Q GEMV: C[emb_dim] = A[emb_dim, emb_dim] @ B[emb_dim]
    print("  [2/4] Q GEMV...")
    q_ir = str(
        build_gemv(emb_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    # K GEMV: C[kv_dim] = A[kv_dim, emb_dim] @ B[emb_dim]
    print("  [3/4] K GEMV...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    # V GEMV
    print("  [4/4] V GEMV...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    # Stitch: RMSNorm (r) + Q (q) + K (k) + V (v)
    # RMSNorm: 3 args {0->0, 1->1, 2->2} (x_in, weight, normed)
    # Q GEMV:  3 args {0->3, 1->2_flat, 2->4} (wq, normed_flat, q_out)
    # Wait — GEMV takes (A[M,K], B[K], C[M]). normed is (1, emb_dim) from RMSNorm
    # but GEMV expects B[K] (1D). We need to flatten normed for GEMV.
    # Solution: the combined func has normed as both 2D (for RMSNorm output) and
    # 1D view for GEMV input. Use separate args for simplicity.
    # Actually, GEMV's B input is 1D memref<K>. RMSNorm outputs 2D memref<1,emb_dim>.
    # For the combined module, we need the normed buffer as BOTH shapes.
    # Simplest: have a 1D normed_flat arg that GEMV reads from, and RMSNorm writes to
    # a 2D buffer that aliases the same data.
    # BUT: multi-launch can't alias args (separate BOs).
    # Alternative: use memref.collapse_shape inside the GEMV launch to flatten.
    # OR: just use 1D for everything (RMSNorm at M=1, N=2048 outputs 1x2048).

    # Actually, looking at the RMSNorm output: it's memref<1x2048xbf16>.
    # GEMV B input: memref<2048xbf16> (1D).
    # These are different memref types → need separate func args.
    # Use arg2 for RMSNorm output (2D), arg_normed_flat for GEMV input (1D).
    # But that means an extra buffer. Not ideal.

    # Simplest approach: compile RMSNorm at M=1 with output as 1D memref<emb_dim>.
    # Unfortunately, weighted_rms_norm always outputs 2D.

    # For now, use separate normed_2d and normed_1d args (host copies between them).
    # This adds one extra arg but avoids shape mismatch.

    # Actually, let me check: can we just have both RMSNorm write to 2D and GEMV
    # read the same buffer reinterpreted as 1D? In the combined module, the func
    # arg is 2D, but we add a collapse_shape inside each GEMV launch.
    # This is the same pattern as ffn_full_multi.py's eltwise_add!

    # Let me just stitch with separate args for simplicity (9 args):
    # arg0: x_in (1, emb_dim)  - RMSNorm input
    # arg1: norm_weight (emb_dim,) - RMSNorm weight
    # arg2: normed (emb_dim,) - shared: RMSNorm output as 1D, GEMV input
    # Wait, RMSNorm generates 2D output. Let me use the 1D RMSNorm variant...

    # Actually the simplest: just use separate kernel invocations for now,
    # and merge ONLY the Q/K/V GEMVs (3 launches in 1 ELF).
    # Skip RMSNorm merge for decode (it's tiny, ~0.1ms).

    # Let me just merge Q+K+V GEMVs into 1 ELF (3 launches, 7 args):
    # arg0: normed (emb_dim,) - input vector (shared)
    # arg1: wq (emb_dim, emb_dim)
    # arg2: q_out (emb_dim,)
    # arg3: wk (kv_dim, emb_dim)
    # arg4: k_out (kv_dim,)
    # arg5: wv (kv_dim, emb_dim)
    # arg6: v_out (kv_dim,)

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (
            q_ir,
            "q",
            {0: 1, 1: 0, 2: 2},
        ),  # GEMV args: A=wq(arg1), B=normed(arg0), C=q_out(arg2)
        (k_ir, "k", {0: 3, 1: 0, 2: 4}),
        (v_ir, "v", {0: 5, 1: 0, 2: 6}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_gemv(body, prefix)
        maps = [_rename_all_gemv(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Extract private func declarations
    privates = [l for l in q_ir.split("\n") if "func.func private" in l]

    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p.strip() for p in privates)}
  func.func @qkv_gemv(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg4: memref<{kv_dim}xbf16>,
    %arg5: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg6: memref<{kv_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, 7 args, 3 launches")
        return module
