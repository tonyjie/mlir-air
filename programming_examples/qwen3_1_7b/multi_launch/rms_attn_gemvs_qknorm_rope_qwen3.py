# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase B Step 1 — fused 8-launch decode ELF for Qwen3:
  RMSNorm 1D + Q/K/V GEMV + Q-Norm + K-Norm + RoPE Q + RoPE K.

Replaces the 4-launch `rms_attn_gemvs.elf` with an 8-launch superset that
includes Q/K Norm and RoPE on NPU.

Func signature (17 args, 8 launches; inputs first, outputs last for
XRTRunner compatibility):

    inputs (9):
      arg0  x_in        memref<emb_dim   xbf16>
      arg1  norm_w      memref<emb_dim   xbf16>
      arg2  wq          memref<q_dim x emb_dim xbf16>
      arg3  wk          memref<kv_dim x emb_dim xbf16>
      arg4  wv          memref<kv_dim x emb_dim xbf16>
      arg5  q_norm_w    memref<head_dim  xbf16>
      arg6  k_norm_w    memref<head_dim  xbf16>
      arg7  lut_q       memref<n_heads*head_dim    xbf16>
      arg8  lut_k       memref<n_kv_heads*head_dim xbf16>

    intermediates (5) — invocation can mark these intermediate_indices:
      arg9   normed     memref<emb_dim xbf16>     (RMSNorm out → GEMV B)
      arg10  q          memref<q_dim   xbf16>     (Q GEMV out → Q-Norm in)
      arg11  k          memref<kv_dim  xbf16>     (K GEMV out → K-Norm in)
      arg12  q_normed   memref<q_dim   xbf16>     (Q-Norm out → RoPE Q in)
      arg13  k_normed   memref<kv_dim  xbf16>     (K-Norm out → RoPE K in)

    outputs (3):
      arg14  v          memref<kv_dim  xbf16>     V (used directly as V)
      arg15  q_roped    memref<q_dim   xbf16>     final Q (post-RoPE)
      arg16  k_roped    memref<kv_dim  xbf16>     final K (post-RoPE)
"""

import os
import re
import sys

from ml_dtypes import bfloat16

_THIS_DIR = os.path.dirname(__file__)
_EXAMPLES = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (
    _EXAMPLES,
    os.path.join(_EXAMPLES, "llama3"),
    os.path.join(_EXAMPLES, "matrix_vector_multiplication", "bf16"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# New helper: per-head RMSNorm with 1D func args (for stitching)
# ---------------------------------------------------------------------------


def _build_qknorm_per_head_1d(n_rows, head_dim, np_dtype, vector_size=16):
    """Build per-head RMSNorm for Q or K (n_rows × head_dim) with 1D-arg func.

    Mirrors `_build_rms_1d` from llama3 rms_gemv_rope_multi (M=1 case) but
    extended to arbitrary n_rows via an outer row loop in the herd body.
    """
    # Import explicit names (wildcard imports only allowed at module level)
    from air.ir import (
        AffineMap,
        AffineMapAttr,
        F32Type,
        IntegerAttr,
        MemRefType,
        VectorType,
    )
    from air.dialects.air import (
        MemorySpace,
        herd,
        launch,
        segment,
        dma_memcpy_nd,
        module_builder,
    )
    from air.dialects.vector import BroadcastOp
    from air.extras import types as T
    from air.dialects.func import FuncOp
    from air.dialects.memref import (
        AllocOp,
        DeallocOp,
        subview,
        expand_shape as memref_expand_shape,
    )
    from air.dialects.scf import for_, yield_
    from air.dialects.vector import transfer_read, transfer_write
    from air.dialects import arith, math as math_dialect
    from air.backend.xrt_runner import type_mapper

    try:
        from air.dialects.vector import reduction as vector_reduction
    except ImportError:
        from air.dialects.vector import vector_reduction

    range_ = for_
    EPS = 1e-6
    n = head_dim
    assert n % vector_size == 0
    total = n_rows * n

    @module_builder
    def _builder():
        # Type constructors require an active MLIR Context — must be inside @module_builder.
        xrt_dtype = type_mapper(np_dtype)
        vecTy = VectorType.get([vector_size], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        l3_1d_ty = MemRefType.get([total], xrt_dtype)
        l3_2d_ty = MemRefType.get([n_rows, n], xrt_dtype)
        l3_w_ty = MemRefType.get([n], xrt_dtype)

        l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1RowTy = MemRefType.get([n], xrt_dtype, memory_space=l1_mem_space)
        l1VecTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1_mem_space)

        @FuncOp.from_py_func(l3_1d_ty, l3_w_ty, l3_1d_ty)
        def qknorm_1d(x_1d, weight, out_1d):
            @launch(operands=[x_1d, weight, out_1d])
            def qkn_launch(l_x_1d, l_weight, l_out_1d):
                l_x_2d = memref_expand_shape(
                    l3_2d_ty, l_x_1d, [[0, 1]], [], [n_rows, n]
                )
                l_out_2d = memref_expand_shape(
                    l3_2d_ty, l_out_1d, [[0, 1]], [], [n_rows, n]
                )

                @segment(name="qkn_seg", operands=[l_x_2d, l_weight, l_out_2d])
                def qkn_seg(s_x_2d, s_weight, s_out_2d):
                    @herd(
                        name="qkn_herd",
                        sizes=[1, 1],
                        operands=[s_x_2d, s_weight, s_out_2d],
                    )
                    def qkn_body(_tx, _ty, _sx, _sy, l3_in, l3_weight, l3_out):
                        l1_row = AllocOp(l1RowTy, [], [])
                        l1_out = AllocOp(l1RowTy, [], [])
                        l1_weight_buf = AllocOp(l1RowTy, [], [])
                        l1_acc = AllocOp(l1VecTy, [], [])

                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        n_f = arith.ConstantOp(xrt_dtype, float(n))
                        eps_f = arith.ConstantOp(xrt_dtype, EPS)
                        v_zero = BroadcastOp(vecTy, cst0)

                        # DMA weight to L1 once, reused across rows.
                        dma_memcpy_nd(l1_weight_buf, l3_weight)

                        for row_iv in range_(0, n_rows, 1):
                            dma_memcpy_nd(
                                l1_row,
                                l3_in,
                                src_offsets=[row_iv, 0],
                                src_sizes=[1, n],
                                src_strides=[n, 1],
                            )

                            # Sum of x^2
                            transfer_write(
                                None, v_zero, l1_acc, [c0], identity_map, [True]
                            )
                            for j in range_(0, n, vector_size):
                                sub_row = subview(
                                    l1_row.result, [j], [vector_size], [1]
                                )
                                sub_tmp = subview(
                                    l1_out.result, [j], [vector_size], [1]
                                )
                                v_x = transfer_read(
                                    vecTy, sub_row, [c0], identity_map, cst0, [True]
                                )
                                v_sq = arith.mulf(v_x, v_x)
                                transfer_write(
                                    None, v_sq, sub_tmp, [c0], identity_map, [True]
                                )
                                v_sq_rd = transfer_read(
                                    vecTy, sub_tmp, [c0], identity_map, cst0, [True]
                                )
                                v_acc = transfer_read(
                                    vecTy, l1_acc, [c0], identity_map, cst0, [True]
                                )
                                v_sum = arith.addf(v_acc, v_sq_rd)
                                transfer_write(
                                    None, v_sum, l1_acc, [c0], identity_map, [True]
                                )
                                yield_([])

                            v_final = transfer_read(
                                vecTy, l1_acc, [c0], identity_map, cst0, [True]
                            )
                            total_sum = vector_reduction(xrt_dtype, "add", v_final)
                            rms = arith.divf(total_sum, n_f)

                            f32 = F32Type.get()
                            rms_eps = arith.addf(rms, eps_f)
                            rms_eps_f32 = arith.extf(f32, rms_eps)
                            rstd_f32 = math_dialect.rsqrt(rms_eps_f32)
                            rstd = arith.truncf(xrt_dtype, rstd_f32)

                            v_rstd = BroadcastOp(vecTy, rstd)
                            for j in range_(0, n, vector_size):
                                sub_row = subview(
                                    l1_row.result, [j], [vector_size], [1]
                                )
                                sub_w = subview(
                                    l1_weight_buf.result, [j], [vector_size], [1]
                                )
                                sub_out = subview(
                                    l1_out.result, [j], [vector_size], [1]
                                )
                                v_x = transfer_read(
                                    vecTy, sub_row, [c0], identity_map, cst0, [True]
                                )
                                v_w = transfer_read(
                                    vecTy, sub_w, [c0], identity_map, cst0, [True]
                                )
                                v_normed = arith.mulf(v_x, v_rstd)
                                v_weighted = arith.mulf(v_normed, v_w)
                                transfer_write(
                                    None,
                                    v_weighted,
                                    sub_out,
                                    [c0],
                                    identity_map,
                                    [True],
                                )
                                yield_([])

                            dma_memcpy_nd(
                                l3_out,
                                l1_out,
                                dst_offsets=[row_iv, 0],
                                dst_sizes=[1, n],
                                dst_strides=[n, 1],
                            )
                            yield_([])

                        DeallocOp(l1_row)
                        DeallocOp(l1_out)
                        DeallocOp(l1_weight_buf)
                        DeallocOp(l1_acc)

    return _builder()


# ---------------------------------------------------------------------------
# Stitching helpers (mirrors rms_attn_gemvs_qwen3.py)
# ---------------------------------------------------------------------------


def _rename_all_for_qwen3(text, prefix):
    """Preserve matvec, linalg_fill, AND rope extern symbols across rename."""
    extern = {
        "@matvec_vectorized_bf16_bf16",
        "@linalg_fill_bf16",
        "@rope",
    }
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


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rms_attn_gemvs_qknorm_rope_qwen3_module(
    emb_dim=1024,
    q_dim=2048,
    kv_dim=1024,
    n_heads=16,
    n_kv_heads=8,
    head_dim=128,
    tile_m=8,
    m_input=4,
    herd_m=8,
    rope_herd_x=8,
):
    """Build the 8-launch fused decode ELF for Qwen3."""
    from llama3.multi_launch_builder.rms_gemv_rope_multi import (
        _build_rms_1d,
        _build_rope_1d,
    )
    from llama3.multi_launch_builder.superseded.ffn_full_multi import (
        _extract_between_func_and_return,
        _fix_launch_func_args,
    )
    from matvec import build_module as build_gemv

    print(f"  [1/8] RMSNorm 1D (M=1, N={emb_dim})...")
    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, vector_size=16))

    print(f"  [2/8] Q GEMV (m={q_dim}, k={emb_dim})...")
    q_ir = str(build_gemv(q_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [3/8] K GEMV (m={kv_dim}, k={emb_dim})...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [4/8] V GEMV (m={kv_dim}, k={emb_dim})...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [5/8] Q-Norm per-head (n_rows={n_heads}, head_dim={head_dim})...")
    qnorm_ir = str(_build_qknorm_per_head_1d(n_heads, head_dim, bfloat16))

    print(f"  [6/8] K-Norm per-head (n_rows={n_kv_heads}, head_dim={head_dim})...")
    knorm_ir = str(_build_qknorm_per_head_1d(n_kv_heads, head_dim, bfloat16))

    print(f"  [7/8] RoPE Q (n_rows={n_heads}, head_dim={head_dim})...")
    rope_q_ir = str(_build_rope_1d(n_heads, head_dim, bfloat16, herd_x=rope_herd_x))

    print(f"  [8/8] RoPE K (n_rows={n_kv_heads}, head_dim={head_dim})...")
    rope_k_ir = str(_build_rope_1d(n_kv_heads, head_dim, bfloat16, herd_x=rope_herd_x))

    def _extract_affine_maps(ir_text):
        return [l for l in ir_text.split("\n") if l.startswith("#")]

    # Combined arg layout (inputs first, intermediates middle, outputs last):
    #   0 x_in, 1 norm_w, 2 wq, 3 wk, 4 wv,
    #   5 q_norm_w, 6 k_norm_w, 7 lut_q, 8 lut_k,
    #   9 normed, 10 q, 11 k, 12 q_normed, 13 k_normed,
    #   14 v, 15 q_roped, 16 k_roped
    #
    # Sub-kernel arg map (sub-arg → combined arg):
    #   RMSNorm 1D    (x_1d, weight, out_1d):     {0:0, 1:1, 2:9}
    #   Q GEMV        (A=wq, B=normed, C=q):      {0:2, 1:9, 2:10}
    #   K GEMV        (A=wk, B=normed, C=k):      {0:3, 1:9, 2:11}
    #   V GEMV        (A=wv, B=normed, C=v):      {0:4, 1:9, 2:14}
    #   Q-Norm        (x_1d=q, weight=q_norm_w, out_1d=q_normed): {0:10, 1:5, 2:12}
    #   K-Norm        (x_1d=k, weight=k_norm_w, out_1d=k_normed): {0:11, 1:6, 2:13}
    #   RoPE Q        (in=q_normed, lut=lut_q, out=q_roped):       {0:12, 1:7, 2:15}
    #   RoPE K        (in=k_normed, lut=lut_k, out=k_roped):       {0:13, 1:8, 2:16}
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "rms", {0: 0, 1: 1, 2: 9}),
        (q_ir, "q", {0: 2, 1: 9, 2: 10}),
        (k_ir, "k", {0: 3, 1: 9, 2: 11}),
        (v_ir, "v", {0: 4, 1: 9, 2: 14}),
        (qnorm_ir, "qn", {0: 10, 1: 5, 2: 12}),
        (knorm_ir, "kn", {0: 11, 1: 6, 2: 13}),
        (rope_q_ir, "rq", {0: 12, 1: 7, 2: 15}),
        (rope_k_ir, "rk", {0: 13, 1: 8, 2: 16}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_for_qwen3(body, prefix)
        maps = [_rename_all_for_qwen3(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    privates = set()
    for ir in (rms_ir, q_ir, k_ir, v_ir, qnorm_ir, knorm_ir, rope_q_ir, rope_k_ir):
        for line in ir.split("\n"):
            if "func.func private" in line:
                privates.add(line.strip())
    privates_str = "\n  ".join(sorted(privates))

    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @rms_attn_gemvs_qknorm_rope(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{q_dim}x{emb_dim}xbf16>,
    %arg3: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg4: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg5: memref<{head_dim}xbf16>,
    %arg6: memref<{head_dim}xbf16>,
    %arg7: memref<{n_heads * head_dim}xbf16>,
    %arg8: memref<{n_kv_heads * head_dim}xbf16>,
    %arg9: memref<{emb_dim}xbf16>,
    %arg10: memref<{q_dim}xbf16>,
    %arg11: memref<{kv_dim}xbf16>,
    %arg12: memref<{q_dim}xbf16>,
    %arg13: memref<{kv_dim}xbf16>,
    %arg14: memref<{kv_dim}xbf16>,
    %arg15: memref<{q_dim}xbf16>,
    %arg16: memref<{kv_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
    print(f"  Module: {len(combined.splitlines())} lines, 17 args, 8 launches")
    return module


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dim", type=int, default=1024)
    parser.add_argument("--q-dim", type=int, default=2048)
    parser.add_argument("--kv-dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("-p", "--print-module-only", action="store_true")
    args = parser.parse_args()

    print(
        f"Phase B Step 1: rms_attn_gemvs_qknorm_rope_qwen3 standalone "
        f"(emb_dim={args.emb_dim}, q_dim={args.q_dim}, kv_dim={args.kv_dim}, "
        f"n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim})"
    )

    # Need rope.o + mv.o on disk before XRTBackend can link.
    from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(head_dim=args.head_dim)

    module = build_rms_attn_gemvs_qknorm_rope_qwen3_module(
        emb_dim=args.emb_dim,
        q_dim=args.q_dim,
        kv_dim=args.kv_dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    # ---- Random inputs ----
    np.random.seed(42)
    eps = 1e-6
    head_dim = args.head_dim
    half = head_dim // 2
    x_in = (np.random.randn(args.emb_dim) * 1.0).astype(bfloat16)
    norm_w = (np.random.randn(args.emb_dim) * 0.1 + 1.0).astype(bfloat16)
    wq = (np.random.randn(args.q_dim, args.emb_dim) * 0.02).astype(bfloat16)
    wk = (np.random.randn(args.kv_dim, args.emb_dim) * 0.02).astype(bfloat16)
    wv = (np.random.randn(args.kv_dim, args.emb_dim) * 0.02).astype(bfloat16)
    q_norm_w = (np.random.randn(head_dim) * 0.5 + 1.0).astype(bfloat16)
    k_norm_w = (np.random.randn(head_dim) * 0.5 + 1.0).astype(bfloat16)
    # Half-split LUT row, repeated per head
    cos_v = np.cos(np.arange(half, dtype=np.float64) / 100.0)
    sin_v = np.sin(np.arange(half, dtype=np.float64) / 100.0)
    lut_row = np.concatenate([cos_v, sin_v]).astype(bfloat16)
    lut_q = np.tile(lut_row, args.n_heads).astype(bfloat16)
    lut_k = np.tile(lut_row, args.n_kv_heads).astype(bfloat16)

    # ---- CPU reference ----
    x_f32 = x_in.astype(np.float32)
    w_f32 = norm_w.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32) + eps)
    normed_ref = ((x_f32 / rms) * w_f32).astype(bfloat16)
    q_ref = (wq.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)
    k_ref = (wk.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)
    v_ref = (wv.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)

    def _per_head_rms(x_flat, w, n_rows):
        x = x_flat.astype(np.float32).reshape(n_rows, head_dim)
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return ((x / rms) * w.astype(np.float32)).astype(bfloat16).reshape(-1)

    q_normed_ref = _per_head_rms(q_ref, q_norm_w, args.n_heads)
    k_normed_ref = _per_head_rms(k_ref, k_norm_w, args.n_kv_heads)

    def _rope_per_head(x_flat, n_rows):
        x = x_flat.astype(np.float32).reshape(n_rows, head_dim)
        cos = cos_v.astype(np.float32)
        sin = sin_v.astype(np.float32)
        x1, x2 = x[:, :half], x[:, half:]
        out = np.empty_like(x)
        out[:, :half] = x1 * cos - x2 * sin
        out[:, half:] = x1 * sin + x2 * cos
        return out.astype(bfloat16).reshape(-1)

    q_roped_ref = _rope_per_head(q_normed_ref, args.n_heads)
    k_roped_ref = _rope_per_head(k_normed_ref, args.n_kv_heads)

    # ---- Compile + run on NPU2 ----
    from air.backend.xrt import XRTBackend
    import filelock
    import pyxrt as xrt

    print("\nCompiling + running on NPU2...")
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
        runtime_loop_tiling_sizes=[4, 4],
        use_lock_race_condition_fix=True,
        output_format="elf",
        instance_name="rms_attn_gemvs_qknorm_rope",
    )
    t0 = time.time()
    artifact = backend.compile(module)
    print(f"  Compile: {time.time()-t0:.1f}s")

    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)

    normed_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    q_buf = np.zeros(args.q_dim, dtype=bfloat16)
    k_buf = np.zeros(args.kv_dim, dtype=bfloat16)
    q_normed_buf = np.zeros(args.q_dim, dtype=bfloat16)
    k_normed_buf = np.zeros(args.kv_dim, dtype=bfloat16)
    v_buf = np.zeros(args.kv_dim, dtype=bfloat16)
    q_roped_buf = np.zeros(args.q_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(args.kv_dim, dtype=bfloat16)

    inputs_all = [
        x_in,
        norm_w,
        wq,
        wk,
        wv,
        q_norm_w,
        k_norm_w,
        lut_q,
        lut_k,
        normed_buf,
        q_buf,
        k_buf,
        q_normed_buf,
        k_normed_buf,
        v_buf,
        q_roped_buf,
        k_roped_buf,
    ]
    sizes = [a.size * a.itemsize for a in inputs_all]
    bos = [xrt.ext.bo(backend.device, s) for s in sizes]
    for i, a in enumerate(inputs_all):
        bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
        bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = xrt.run(backend.kernel)
    for i, bo in enumerate(bos):
        run.set_arg(i, bo)
    t1 = time.time()
    run.start()
    run.wait2()
    print(f"  NPU run: {(time.time()-t1)*1000:.2f} ms")

    for idx in (9, 10, 11, 12, 13, 14, 15, 16):
        bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _read(idx, shape):
        return bos[idx].read(sizes[idx], 0).view(np.int16).view(bfloat16).reshape(shape)

    npu_normed = _read(9, (args.emb_dim,))
    npu_q = _read(10, (args.q_dim,))
    npu_k = _read(11, (args.kv_dim,))
    npu_q_normed = _read(12, (args.q_dim,))
    npu_k_normed = _read(13, (args.kv_dim,))
    npu_v = _read(14, (args.kv_dim,))
    npu_q_roped = _read(15, (args.q_dim,))
    npu_k_roped = _read(16, (args.kv_dim,))
    backend.unload()

    def _cos(a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    print("\nCorrectness vs CPU reference:")
    cs = {
        "normed": _cos(npu_normed, normed_ref),
        "q": _cos(npu_q, q_ref),
        "k": _cos(npu_k, k_ref),
        "q_normed": _cos(npu_q_normed, q_normed_ref),
        "k_normed": _cos(npu_k_normed, k_normed_ref),
        "v": _cos(npu_v, v_ref),
        "q_roped": _cos(npu_q_roped, q_roped_ref),
        "k_roped": _cos(npu_k_roped, k_roped_ref),
    }
    for name, c in cs.items():
        marker = "PASS" if c > 0.99 else "FAIL"
        print(f"  [{marker}] {name:9s} cosine={c:.6f}")
    sys.exit(0 if all(c > 0.99 for c in cs.values()) else 1)
