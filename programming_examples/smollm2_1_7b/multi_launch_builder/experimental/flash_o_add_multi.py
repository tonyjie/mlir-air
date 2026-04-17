#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlashAttention + O Projection + Residual Add — 3-launch multi-launch ELF.

Builds a single AIR function with 3 sequential air.launch operations:
  1. FlashAttention  [8,16]  q_roped x k_roped x v -> attn_out
  2. O GEMM          [8,4]   attn_out x wo -> proj
  3. Residual Add    [8,1]   proj + x_residual -> output (1D)

8 func args (3 launches). Merges the flash_attn and o_proj_add kernels
into a single XRT invocation, eliminating one dispatch overhead per layer.

Usage:
    python3 flash_o_add_multi.py -p           # print combined MLIR
    python3 flash_o_add_multi.py              # compile + run + validate
"""

import argparse
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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

from _llm_shared.kernel_builder.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _fix_launch_func_args,
    _rename_all_with_externs,
)

range_ = for_

# FlashAttention external kernel functions (must NOT be renamed)
_EXTERN_FUNCS = {
    "@matmul_bf16",
    "@zero_fill_g_bf16",
    "@zero_fill_gp_bf16",
    "@zero_fill_sp_bf16",
    "@neg_inf_fill_up_bf16",
    "@matmul_a_b_bf16",
    "@matmul_g_b_bf16",
    "@fused_softmax",
    "@maximum_up_u_bf16",
    "@exp_up_minus_u",
    "@mul_r_gp",
    "@accum_sp_r_s",
    "@vector_copy_32elems",
    "@copy_tile",
    "@div_gp_sp",
    "@add_gp_g",
    "@apply_causal_mask",
}


def _extract_affine_sets(ir_text):
    """Extract top-level #set declarations (affine integer sets)."""
    return [l for l in ir_text.split("\n") if l.startswith("#set")]


def _rename_sets(text, prefix):
    """Rename #set references: #set -> #prefix_set, #set1 -> #prefix_set1."""
    for name in sorted(set(re.findall(r"#set\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)
    return text


def _extract_channel_decls(ir_text):
    """Extract air.channel declarations (module-level, not put/get ops)."""
    result = []
    for line in ir_text.split("\n"):
        stripped = line.strip()
        if (
            stripped.startswith("air.channel @")
            and "air.channel.put" not in stripped
            and "air.channel.get" not in stripped
        ):
            result.append(stripped)
    return result


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_flash_o_add_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # O GEMM tile config
    tile_m=64,
    tile_k_l2=256,
    tile_k_l1=32,
    tile_n=64,
    herd_m=8,
    herd_n=4,
    # Add config
    add_herd_x=8,
    add_herd_y=1,
    add_vector_size=16,
    print_kernels=False,
):
    """Build 3-launch module: FlashAttention + O GEMM + Residual Add.

    Returns:
        Module with func @flash_o_add and 8 memref args:
            %arg0: q_roped      (seq_len, emb_dim)      FlashAttn Q input
            %arg1: k_roped      (seq_len, kv_dim)        FlashAttn K input
            %arg2: v            (seq_len, kv_dim)         FlashAttn V input
            %arg3: attn_out     (seq_len, emb_dim)       FlashAttn output / O GEMM input
            %arg4: wo           (emb_dim, emb_dim)       O GEMM weight
            %arg5: proj         (seq_len, emb_dim)       O GEMM output / Add input
            %arg6: x_residual   (seq_len, emb_dim)       Residual skip connection
            %arg7: output       (n_total,)               1D final output
    """
    from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module

    n_total = seq_len * emb_dim
    add_tile_n = emb_dim

    # ---- Build sub-kernels ----

    # 1. FlashAttention
    print(f"  [1/3] FlashAttention (lq={seq_len}, heads={n_heads}/{n_kv_heads})...")
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )

    attn_mod = build_attn(
        lk=seq_len,
        lkp=head_dim,
        lq=seq_len,
        lqp=256,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=4,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=True,
    )
    attn_ir = str(attn_mod)

    # 2. O GEMM
    print("  [2/3] O GEMM...")
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

    # 3. Eltwise Add (2D inputs, 1D output, collapse_shape inside launch)
    print("  [3/3] Residual Add...")

    @module_builder
    def _build_add_2d():
        from air.dialects.memref import collapse_shape as memref_collapse_shape

        xrt_dtype = type_mapper(bfloat16)
        l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        l3_1d_ty = MemRefType.get([n_total], xrt_dtype)
        total_tiles = add_herd_x * add_herd_y
        chunk_size = n_total // total_tiles
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
        for name, ir in [
            ("FlashAttention", attn_ir),
            ("O GEMM", gemm_ir),
            ("Residual Add", add_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping:
    #   FlashAttn: {0:0, 1:1, 2:2, 3:3}  (q_roped, k_roped, v, attn_out)
    #   O GEMM:    {0:3, 1:4, 2:5}        (attn_out, wo, proj)
    #   Res Add:   {0:5, 1:6, 2:7}        (proj, x_residual, output)

    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (attn_ir, "fa", {0: 0, 1: 1, 2: 2, 3: 3}),
        (gemm_ir, "og", {0: 3, 1: 4, 2: 5}),
        (add_ir, "ra", {0: 5, 1: 6, 2: 7}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        sets = _extract_affine_sets(ir)
        body = _rename_all_with_externs(body, prefix, _EXTERN_FUNCS)
        body = _rename_sets(body, prefix)
        maps = [_rename_all_with_externs(m, prefix, _EXTERN_FUNCS) for m in maps]
        sets = [_rename_sets(s, prefix) for s in sets]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)
        maps_all.extend(sets)

    # Collect private func declarations (FlashAttention has 16)
    all_privates = set()
    for p in _extract_private_funcs(attn_ir):
        all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    # Extract and rename FlashAttention channel declarations
    channel_decls = _extract_channel_decls(attn_ir)
    renamed_channels = []
    for decl in channel_decls:
        renamed = _rename_all_with_externs(decl, "fa", _EXTERN_FUNCS)
        renamed_channels.append(renamed)
    channels_str = "\n  ".join(renamed_channels)

    # Assemble (8 func args, 3 launches)
    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  {channels_str}
  func.func @flash_o_add(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{seq_len}x{kv_dim}xbf16>,
    %arg2: memref<{seq_len}x{kv_dim}xbf16>,
    %arg3: memref<{seq_len}x{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg5: memref<{seq_len}x{emb_dim}xbf16>,
    %arg6: memref<{seq_len}x{emb_dim}xbf16>,
    %arg7: memref<{n_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
    return
  }}
}}
"""

    with Context() as ctx:
        try:
            module = Module.parse(combined, ctx)
        except Exception:
            with open("/tmp/debug_flash_o_add.mlir", "w") as f:
                f.write(combined)
            print("  PARSE ERROR: dumped to /tmp/debug_flash_o_add.mlir")
            raise
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEQ_LEN = 2048
    EMB_DIM = 2048
    KV_DIM = 512
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(
        description="FlashAttention + O GEMM + Residual Add multi-launch test"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    print(f"Flash+O+Add Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, kv={KV_DIM}")

    module = build_flash_o_add_module(
        seq_len=SEQ_LEN,
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="flash_o_add",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run ----
    np.random.seed(42)

    q_roped = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    k_roped = np.random.uniform(-1.0, 1.0, (SEQ_LEN, KV_DIM)).astype(bfloat16)
    v = np.random.uniform(-1.0, 1.0, (SEQ_LEN, KV_DIM)).astype(bfloat16)
    wo = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    x_residual = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)

    # CPU reference
    print("Computing CPU reference...")

    def attention_ref(q, k, v, n_heads, n_kv_heads, head_dim):
        seq_len = q.shape[0]
        q_f32 = q.astype(np.float32)
        k_f32 = k.astype(np.float32)
        v_f32 = v.astype(np.float32)
        out = np.zeros((seq_len, n_heads * head_dim), dtype=np.float32)
        for h in range(n_heads):
            kv_h = h // (n_heads // n_kv_heads)
            qh = q_f32[:, h * head_dim : (h + 1) * head_dim]
            kh = k_f32[:, kv_h * head_dim : (kv_h + 1) * head_dim]
            vh = v_f32[:, kv_h * head_dim : (kv_h + 1) * head_dim]
            scores = qh @ kh.T / np.sqrt(head_dim)
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
            scores += mask
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn /= attn.sum(axis=-1, keepdims=True)
            out[:, h * head_dim : (h + 1) * head_dim] = attn @ vh
        return out.astype(bfloat16)

    attn_ref = attention_ref(q_roped, k_roped, v, N_HEADS, N_KV_HEADS, HEAD_DIM)
    proj_ref = (attn_ref.astype(np.float32) @ wo.astype(np.float32)).astype(bfloat16)
    output_ref = (proj_ref.astype(np.float32) + x_residual.astype(np.float32)).astype(
        bfloat16
    )

    # Buffers
    attn_out_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    proj_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="flash_o_add",
    )

    # 7 inputs (args 0-6) + 1 expected output (arg 7)
    exit(
        runner.run_test(
            module,
            inputs=[
                q_roped,  # arg0
                k_roped,  # arg1
                v,  # arg2
                attn_out_buf,  # arg3
                wo,  # arg4
                proj_buf,  # arg5
                x_residual,  # arg6
            ],
            expected_outputs=[
                output_ref.flatten(),  # arg7 (1D)
            ],
            rtol=0.2,
            atol=4.0,
            min_correlation=0.95,
        )
    )
