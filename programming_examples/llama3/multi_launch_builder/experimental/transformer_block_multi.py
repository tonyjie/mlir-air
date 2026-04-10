#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Full Transformer Block -- 15-launch multi-launch ELF.

Builds a single AIR function with 15 sequential air.launch operations
implementing one complete LLaMA-3.2-1B transformer layer:

  L1:  RMSNorm      [8,1]   x_in x attn_norm_w -> normed
  L2:  Q GEMM       [8,4]   normed x wq -> q
  L3:  K GEMM       [8,4]   normed x wk -> k
  L4:  V GEMM       [8,4]   normed x wv -> v
  L5:  RoPE Q       [8,1]   q(collapse) x lut_q -> q_roped(collapse)
  L6:  RoPE K       [8,1]   k(collapse) x lut_k -> k_roped(collapse)
  L7:  FlashAttn    [8,16]  q_roped, k_roped, v -> attn_out
  L8:  O GEMM       [8,4]   attn_out x wo -> proj
  L9:  Residual Add [8,1]   proj(collapse) + x_in(collapse) -> res1(collapse)
  L10: FFN RMSNorm  [8,1]   res1 x ffn_norm_w -> normed2
  L11: Gate GEMM    [8,4]   normed2 x w_gate -> gate
  L12: Up GEMM      [8,4]   normed2 x w_up -> up
  L13: SwiGLU       [8,1]   SiLU(gate) x up -> swiglu
  L14: Down GEMM    [8,4]   swiglu x w_down -> down
  L15: FFN Add      [8,1]   down(collapse) + res1(collapse) -> output(1D)

27 func args (15 launches). All intermediates are passed as func args so
that each launch can be independently tested. Collapse shapes are used
inside launches for 2D<->1D aliasing (RoPE, eltwise adds).

Usage:
    python3 transformer_block_multi.py -p           # print combined MLIR
    python3 transformer_block_multi.py              # compile + run + validate
"""

import argparse
import os
import sys

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

from llama3.multi_launch_builder.ffn_full_multi import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)
from llama3.multi_launch_builder.rms_gemms_rope_multi import (
    _build_rope_2d,
    _rename_all_with_externs,
)

range_ = for_

# External kernel functions that must NOT be renamed during stitching.
_EXTERN_FUNCS = {
    # RMSNorm
    "@zero_vectorized_bf16",
    # GEMMs
    "@matmul_bf16",
    # RoPE
    "@rope",
    # SwiGLU
    "@silu_and_mul_bf16",
    # FlashAttention (16 functions from attn_npu2.o)
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


# ---------------------------------------------------------------------------
# 2D eltwise add builder (all three args 2D, collapse inside launch)
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_2d(rows, cols, np_dtype, vector_size, herd_x, herd_y):
    """Eltwise add with 2D in/out args: a_2d + b_2d -> out_2d.

    All three func args are 2D memrefs. Inside the launch, collapse_shape
    flattens to 1D for the DMA + vectorized add. The output remains 2D at
    the func level so the next launch can read it as a 2D memref.
    """
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_2d_ty)
    def eltwise_add_2d_to_2d(arg0_2d, arg1_2d, arg2_2d):
        @launch(operands=[arg0_2d, arg1_2d, arg2_2d])
        def add_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])
            out_flat = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="add_seg", operands=[a_flat, b_flat, out_flat])
            def add_seg(s_a, s_b, s_out):
                total_tiles = herd_x * herd_y
                tile_n = cols
                l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                l1TileTy = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
                vecTy = VectorType.get([vector_size], xrt_dtype)
                identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

                @herd(
                    name="add_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_a, s_b, s_out],
                )
                def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1TileTy, [], [])
                    l1_b = AllocOp(l1TileTy, [], [])
                    l1_out = AllocOp(l1TileTy, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    chunk_size = n // total_tiles
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
                                            AffineConstantExpr.get(herd_y),
                                        ),
                                        AffineSymbolExpr.get(2),
                                    ),
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )

                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )

                        for j in range_(0, tile_n, vector_size):
                            sub_a = subview(l1_a.result, [j], [vector_size], [1])
                            sub_b = subview(l1_b.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_a = transfer_read(
                                vecTy, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vecTy, sub_b, [c0], identity_map, cst0, [True]
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
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


# ---------------------------------------------------------------------------
# 2D->1D eltwise add builder (inputs 2D, output 1D)
# ---------------------------------------------------------------------------


@module_builder
def _build_add_2d_to_1d(rows, cols, np_dtype, vector_size, herd_x, herd_y):
    """Eltwise add: a_2d + b_2d -> out_1d.

    Inputs are 2D memrefs (collapsed to 1D inside launch), output is a 1D
    memref. This is the final layer's residual add producing the flat output.
    """
    from air.dialects.memref import collapse_shape as memref_collapse_shape

    xrt_dtype = type_mapper(np_dtype)
    n = rows * cols
    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
    def eltwise_add_2d_to_1d(arg0_2d, arg1_2d, arg2_1d):
        @launch(operands=[arg0_2d, arg1_2d, arg2_1d])
        def add_launch(l_a, l_b, l_out):
            a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
            b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

            @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
            def add_seg(s_a, s_b, s_out):
                total_tiles = herd_x * herd_y
                tile_n = cols
                l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                l1TileTy = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
                vecTy = VectorType.get([vector_size], xrt_dtype)
                identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

                @herd(
                    name="add_herd",
                    sizes=[herd_x, herd_y],
                    operands=[s_a, s_b, s_out],
                )
                def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                    l1_a = AllocOp(l1TileTy, [], [])
                    l1_b = AllocOp(l1TileTy, [], [])
                    l1_out = AllocOp(l1TileTy, [], [])
                    c0 = arith.ConstantOp.create_index(0)
                    cst0 = arith.ConstantOp(xrt_dtype, 0.0)

                    chunk_size = n // total_tiles
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
                                            AffineConstantExpr.get(herd_y),
                                        ),
                                        AffineSymbolExpr.get(2),
                                    ),
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )

                    for loop_iv in range_(0, chunk_size, tile_n):
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                        dma_memcpy_nd(
                            l1_a,
                            h_a,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_b,
                            h_b,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )

                        for j in range_(0, tile_n, vector_size):
                            sub_a = subview(l1_a.result, [j], [vector_size], [1])
                            sub_b = subview(l1_b.result, [j], [vector_size], [1])
                            sub_out = subview(l1_out.result, [j], [vector_size], [1])
                            v_a = transfer_read(
                                vecTy, sub_a, [c0], identity_map, cst0, [True]
                            )
                            v_b = transfer_read(
                                vecTy, sub_b, [c0], identity_map, cst0, [True]
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
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_a)
                    DeallocOp(l1_b)
                    DeallocOp(l1_out)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_transformer_block_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    hidden_dim=8192,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # Attention GEMM tile config (Q/K/V/O projections)
    attn_tile_m=64,
    attn_tile_k_l2=64,
    attn_tile_k_l1=32,
    attn_tile_n=128,
    attn_herd_m=8,
    attn_herd_n=4,
    # FFN Gate/Up GEMM tile config
    gate_tile_m=64,
    gate_tile_k_l2=64,
    gate_tile_k_l1=32,
    gate_tile_n=128,
    gate_herd_m=8,
    gate_herd_n=4,
    # FFN Down GEMM tile config
    down_tile_m=64,
    down_tile_k_l2=256,
    down_tile_k_l1=32,
    down_tile_n=64,
    down_herd_m=8,
    down_herd_n=4,
    # RoPE config
    rope_herd_x=8,
    # FlashAttention config
    attn_lqp=256,
    attn_num_q_tiles=4,
    attn_num_cascade_stages=4,
    # SwiGLU config
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    # Eltwise add config
    add_herd_x=8,
    add_herd_y=1,
    print_kernels=False,
):
    """Build 15-launch transformer block module.

    Combines: RMSNorm + QKV GEMMs + RoPE Q/K + FlashAttention + O GEMM +
    Residual Add + FFN RMSNorm + Gate/Up GEMMs + SwiGLU + Down GEMM + FFN Add

    Returns:
        Module with func @transformer_block and 27 memref args:
            %arg0:  x_in         memref<seq_len x emb_dim x bf16>    input
            %arg1:  attn_norm_w  memref<emb_dim x bf16>              RMSNorm weight
            %arg2:  normed       memref<seq_len x emb_dim x bf16>    intermediate
            %arg3:  wq           memref<emb_dim x emb_dim x bf16>    weight
            %arg4:  q            memref<seq_len x emb_dim x bf16>    intermediate
            %arg5:  wk           memref<emb_dim x kv_dim x bf16>     weight
            %arg6:  k            memref<seq_len x kv_dim x bf16>     intermediate
            %arg7:  wv           memref<emb_dim x kv_dim x bf16>     weight
            %arg8:  v            memref<seq_len x kv_dim x bf16>     intermediate
            %arg9:  lut_q        memref<q_total x bf16>              1D LUT
            %arg10: lut_k        memref<k_total x bf16>              1D LUT
            %arg11: q_roped      memref<seq_len x emb_dim x bf16>    intermediate
            %arg12: k_roped      memref<seq_len x kv_dim x bf16>     intermediate
            %arg13: attn_out     memref<seq_len x emb_dim x bf16>    intermediate
            %arg14: wo           memref<emb_dim x emb_dim x bf16>    weight
            %arg15: proj         memref<seq_len x emb_dim x bf16>    intermediate
            %arg16: res1         memref<seq_len x emb_dim x bf16>    intermediate
            %arg17: ffn_norm_w   memref<emb_dim x bf16>              weight
            %arg18: normed2      memref<seq_len x emb_dim x bf16>    intermediate
            %arg19: w_gate       memref<emb_dim x hidden_dim x bf16> weight
            %arg20: gate         memref<seq_len x hidden_dim x bf16> intermediate
            %arg21: w_up         memref<emb_dim x hidden_dim x bf16> weight
            %arg22: up           memref<seq_len x hidden_dim x bf16> intermediate
            %arg23: swiglu       memref<seq_len x hidden_dim x bf16> intermediate
            %arg24: w_down       memref<hidden_dim x emb_dim x bf16> weight
            %arg25: down         memref<seq_len x emb_dim x bf16>    intermediate
            %arg26: output       memref<n_total x bf16>              1D final output
    """
    from llama3.llama3_prefill import _build_gemm_module
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )
    from llama3.ffn_swiglu.silu_and_mul import build_module_2d as build_swiglu

    q_total = seq_len * emb_dim
    k_total = seq_len * kv_dim
    n_total = seq_len * emb_dim

    # ---- Build sub-kernels (15 total) ----

    # L1: Attention RMSNorm
    print("  [1/15] RMSNorm (attention)...")
    rms1_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # L2: Q GEMM
    print("  [2/15] Q GEMM...")
    q_gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            attn_tile_m,
            attn_tile_k_l2,
            attn_tile_k_l1,
            attn_tile_n,
            attn_herd_m,
            attn_herd_n,
        )
    )

    # L3: K GEMM
    print("  [3/15] K GEMM...")
    k_gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            attn_tile_m,
            attn_tile_k_l2,
            attn_tile_k_l1,
            attn_tile_n,
            attn_herd_m,
            attn_herd_n,
        )
    )

    # L4: V GEMM
    print("  [4/15] V GEMM...")
    v_gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            attn_tile_m,
            attn_tile_k_l2,
            attn_tile_k_l1,
            attn_tile_n,
            attn_herd_m,
            attn_herd_n,
        )
    )

    # L5: RoPE Q (2D in/out with collapse inside launch)
    print(
        f"  [5/15] RoPE Q (outer={seq_len}x{emb_dim}, "
        f"embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    rope_q_ir = str(_build_rope_2d(seq_len, emb_dim, head_dim, bfloat16, rope_herd_x))

    # L6: RoPE K (2D in/out with collapse inside launch)
    print(
        f"  [6/15] RoPE K (outer={seq_len}x{kv_dim}, "
        f"embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    # L7: FlashAttention (already has air.launch + segment, no wrapping)
    print(
        f"  [7/15] FlashAttention (lq={seq_len}, lk={seq_len}, "
        f"heads={n_heads}/{n_kv_heads}, causal)..."
    )
    attn_mod = build_attn(
        lk=seq_len,
        lkp=head_dim,
        lq=seq_len,
        lqp=attn_lqp,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=attn_num_q_tiles,
        num_cascade_stages=attn_num_cascade_stages,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=True,
    )
    attn_ir = str(attn_mod)

    # L8: O GEMM
    print("  [8/15] O GEMM...")
    o_gemm_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            attn_tile_m,
            attn_tile_k_l2,
            attn_tile_k_l1,
            attn_tile_n,
            attn_herd_m,
            attn_herd_n,
        )
    )

    # L9: Residual Add (2D + 2D -> 2D, collapse inside launch)
    print("  [9/15] Residual Add (2D -> 2D)...")
    res_add_ir = str(
        _build_add_2d_to_2d(seq_len, emb_dim, bfloat16, 16, add_herd_x, add_herd_y)
    )

    # L10: FFN RMSNorm
    print("  [10/15] FFN RMSNorm...")
    rms2_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # L11: Gate GEMM
    print("  [11/15] Gate GEMM...")
    gate_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    # L12: Up GEMM
    print("  [12/15] Up GEMM...")
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    # L13: SwiGLU (2D in/out, collapse inside launch)
    print("  [13/15] SwiGLU...")
    swiglu_mod = build_swiglu(
        seq_len,
        hidden_dim,
        swiglu_tile_n,
        bfloat16,
        herd_x=swiglu_herd_x,
        herd_y=swiglu_herd_y,
    )
    swiglu_ir = str(swiglu_mod)

    # L14: Down GEMM
    print("  [14/15] Down GEMM...")
    down_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            down_tile_m,
            down_tile_k_l2,
            down_tile_k_l1,
            down_tile_n,
            down_herd_m,
            down_herd_n,
        )
    )

    # L15: FFN Add (2D + 2D -> 1D, collapse inside launch)
    print("  [15/15] FFN Add (2D -> 1D)...")
    ffn_add_ir = str(
        _build_add_2d_to_1d(seq_len, emb_dim, bfloat16, 16, add_herd_x, add_herd_y)
    )

    if print_kernels:
        kernel_list = [
            ("L1  RMSNorm", rms1_ir),
            ("L2  Q GEMM", q_gemm_ir),
            ("L3  K GEMM", k_gemm_ir),
            ("L4  V GEMM", v_gemm_ir),
            ("L5  RoPE Q", rope_q_ir),
            ("L6  RoPE K", rope_k_ir),
            ("L7  FlashAttn", attn_ir),
            ("L8  O GEMM", o_gemm_ir),
            ("L9  Res Add", res_add_ir),
            ("L10 FFN RMSNorm", rms2_ir),
            ("L11 Gate GEMM", gate_ir),
            ("L12 Up GEMM", up_ir),
            ("L13 SwiGLU", swiglu_ir),
            ("L14 Down GEMM", down_ir),
            ("L15 FFN Add", ffn_add_ir),
        ]
        for name, ir in kernel_list:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch: extract, rename, remap ----
    #
    # Arg mappings (sub-kernel func arg index -> combined func arg index):
    #   L1  RMSNorm:     {0:0, 1:1, 2:2}
    #   L2  Q GEMM:      {0:2, 1:3, 2:4}
    #   L3  K GEMM:      {0:2, 1:5, 2:6}
    #   L4  V GEMM:      {0:2, 1:7, 2:8}
    #   L5  RoPE Q:      {0:4, 1:9, 2:11}
    #   L6  RoPE K:      {0:6, 1:10, 2:12}
    #   L7  FlashAttn:   {0:11, 1:12, 2:8, 3:13}
    #   L8  O GEMM:      {0:13, 1:14, 2:15}
    #   L9  Res Add:     {0:15, 1:0, 2:16}
    #   L10 FFN RMSNorm: {0:16, 1:17, 2:18}
    #   L11 Gate GEMM:   {0:18, 1:19, 2:20}
    #   L12 Up GEMM:     {0:18, 1:21, 2:22}
    #   L13 SwiGLU:      {0:20, 1:22, 2:23}
    #   L14 Down GEMM:   {0:23, 1:24, 2:25}
    #   L15 FFN Add:     {0:25, 1:16, 2:26}

    stitch_specs = [
        (rms1_ir, "r1", {0: 0, 1: 1, 2: 2}),
        (q_gemm_ir, "qg", {0: 2, 1: 3, 2: 4}),
        (k_gemm_ir, "kg", {0: 2, 1: 5, 2: 6}),
        (v_gemm_ir, "vg", {0: 2, 1: 7, 2: 8}),
        (rope_q_ir, "rq", {0: 4, 1: 9, 2: 11}),
        (rope_k_ir, "rk", {0: 6, 1: 10, 2: 12}),
        (attn_ir, "fa", {0: 11, 1: 12, 2: 8, 3: 13}),
        (o_gemm_ir, "og", {0: 13, 1: 14, 2: 15}),
        (res_add_ir, "ra", {0: 15, 1: 0, 2: 16}),
        (rms2_ir, "r2", {0: 16, 1: 17, 2: 18}),
        (gate_ir, "gg", {0: 18, 1: 19, 2: 20}),
        (up_ir, "ug", {0: 18, 1: 21, 2: 22}),
        (swiglu_ir, "sw", {0: 20, 1: 22, 2: 23}),
        (down_ir, "dg", {0: 23, 1: 24, 2: 25}),
        (ffn_add_ir, "fa2", {0: 25, 1: 16, 2: 26}),
    ]

    def _extract_affine_sets(ir_text):
        """Extract top-level #set declarations (affine integer sets)."""
        return [l for l in ir_text.split("\n") if l.startswith("#set")]

    def _rename_sets(text, prefix):
        """Rename #set references: #set -> #prefix_set, #set1 -> #prefix_set1, etc."""
        import re

        for name in sorted(set(re.findall(r"#set\d*", text)), key=len, reverse=True):
            text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)
        return text

    bodies, maps_all = [], []
    for ir, prefix, arg_map in stitch_specs:
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

    # Collect all private func declarations from sub-kernels that have them
    all_privates = set()
    for ir in [rms1_ir, rope_q_ir, attn_ir, swiglu_ir]:
        for p in _extract_private_funcs(ir):
            all_privates.add(p.strip())
    privates_str = "\n  ".join(sorted(all_privates))

    # Extract and rename channel declarations from FlashAttention
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

    channel_decls = _extract_channel_decls(attn_ir)
    # Rename channel names with the FlashAttention prefix
    renamed_channels = []
    for decl in channel_decls:
        renamed = _rename_all_with_externs(decl, "fa", _EXTERN_FUNCS)
        renamed_channels.append(renamed)
    channels_str = "\n  ".join(renamed_channels)

    # Assemble the combined module (27 func args, 15 launches)
    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  {channels_str}
  func.func @transformer_block(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg6: memref<{seq_len}x{kv_dim}xbf16>,
    %arg7: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg8: memref<{seq_len}x{kv_dim}xbf16>,
    %arg9: memref<{q_total}xbf16>,
    %arg10: memref<{k_total}xbf16>,
    %arg11: memref<{seq_len}x{emb_dim}xbf16>,
    %arg12: memref<{seq_len}x{kv_dim}xbf16>,
    %arg13: memref<{seq_len}x{emb_dim}xbf16>,
    %arg14: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg15: memref<{seq_len}x{emb_dim}xbf16>,
    %arg16: memref<{seq_len}x{emb_dim}xbf16>,
    %arg17: memref<{emb_dim}xbf16>,
    %arg18: memref<{seq_len}x{emb_dim}xbf16>,
    %arg19: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg20: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg21: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg22: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg23: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg24: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg25: memref<{seq_len}x{emb_dim}xbf16>,
    %arg26: memref<{n_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
{bodies[4]}
{bodies[5]}
{bodies[6]}
{bodies[7]}
{bodies[8]}
{bodies[9]}
{bodies[10]}
{bodies[11]}
{bodies[12]}
{bodies[13]}
{bodies[14]}
    return
  }}
}}
"""

    # Debug: dump combined MLIR if parse fails
    with Context() as ctx:
        try:
            module = Module.parse(combined, ctx)
        except Exception as e:
            with open("/tmp/debug_transformer_block.mlir", "w") as f:
                f.write(combined)
            print(
                f"  PARSE ERROR: Combined MLIR dumped to /tmp/debug_transformer_block.mlir"
            )
            raise
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x, weight, eps=1e-5):
    """CPU RMSNorm reference (F32 computation)."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_2d, lut_2d):
    """CPU RoPE reference: interleaved cos/sin encoding."""
    x = x_2d.astype(np.float32)
    lut = lut_2d.astype(np.float32)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * lut[:, 0::2] - x[:, 1::2] * lut[:, 1::2]
    out[:, 1::2] = x[:, 0::2] * lut[:, 1::2] + x[:, 1::2] * lut[:, 0::2]
    return out.astype(bfloat16)


def attention_reference(q, k, v, n_heads, n_kv_heads, head_dim, causal=True):
    """Simple per-head attention reference with GQA and causal mask.

    Args:
        q: (seq_len, n_heads * head_dim) in seq-first layout
        k: (seq_len, n_kv_heads * head_dim) in seq-first layout
        v: (seq_len, n_kv_heads * head_dim) in seq-first layout
        n_heads: number of query heads
        n_kv_heads: number of key/value heads (GQA)
        head_dim: dimension per head
        causal: apply causal mask

    Returns:
        output: (seq_len, n_heads * head_dim)
    """
    seq_len = q.shape[0]
    gqa_group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)

    q_f32 = q.astype(np.float32).reshape(seq_len, n_heads, head_dim)
    k_f32 = k.astype(np.float32).reshape(seq_len, n_kv_heads, head_dim)
    v_f32 = v.astype(np.float32).reshape(seq_len, n_kv_heads, head_dim)

    output = np.zeros((seq_len, n_heads, head_dim), dtype=np.float32)

    for h in range(n_heads):
        kv_h = h // gqa_group_size
        # Q: (seq_len, head_dim), K: (head_dim, seq_len), V: (seq_len, head_dim)
        q_h = q_f32[:, h, :]
        k_h = k_f32[:, kv_h, :]
        v_h = v_f32[:, kv_h, :]

        scores = (q_h @ k_h.T) * scale  # (seq_len, seq_len)

        if causal:
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
            scores = scores + mask

        # Numerically stable softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
        attn_weights = scores_exp / scores_sum

        output[:, h, :] = attn_weights @ v_h

    return output.reshape(seq_len, n_heads * head_dim).astype(bfloat16)


def transformer_block_reference(
    x_in,
    attn_norm_w,
    wq,
    wk,
    wv,
    lut_q,
    lut_k,
    wo,
    ffn_norm_w,
    w_gate,
    w_up,
    w_down,
    n_heads,
    n_kv_heads,
    head_dim,
    eps=1e-5,
):
    """Full transformer block CPU reference (F32 computation, BF16 output).

    Implements: RMSNorm -> QKV GEMM -> RoPE -> Attention -> O proj ->
    Residual Add -> FFN RMSNorm -> Gate/Up -> SwiGLU -> Down -> FFN Add

    Args:
        x_in: (seq_len, emb_dim) input
        attn_norm_w: (emb_dim,) attention RMSNorm weight
        wq: (emb_dim, emb_dim) Q projection
        wk: (emb_dim, kv_dim) K projection
        wv: (emb_dim, kv_dim) V projection
        lut_q: (n_heads * seq_len, head_dim) RoPE Q LUT
        lut_k: (n_kv_heads * seq_len, head_dim) RoPE K LUT
        wo: (emb_dim, emb_dim) O projection
        ffn_norm_w: (emb_dim,) FFN RMSNorm weight
        w_gate: (emb_dim, hidden_dim) gate projection
        w_up: (emb_dim, hidden_dim) up projection
        w_down: (hidden_dim, emb_dim) down projection
        n_heads: number of query heads
        n_kv_heads: number of KV heads
        head_dim: per-head dimension
        eps: RMSNorm epsilon

    Returns:
        output: (seq_len, emb_dim) final output
    """
    seq_len = x_in.shape[0]
    emb_dim = x_in.shape[1]
    kv_dim = wk.shape[1]

    # 1. Attention RMSNorm
    normed = _rms_norm_ref(x_in, attn_norm_w, eps)

    # 2-4. QKV GEMMs
    normed_f32 = normed.astype(np.float32)
    q = (normed_f32 @ wq.astype(np.float32)).astype(bfloat16)
    k = (normed_f32 @ wk.astype(np.float32)).astype(bfloat16)
    v = (normed_f32 @ wv.astype(np.float32)).astype(bfloat16)

    # 5-6. RoPE Q/K (reshape to seq-first interleaved, apply, reshape back)
    q_flat = q.reshape(seq_len, n_heads, head_dim).reshape(seq_len * n_heads, head_dim)
    q_roped = _rope_ref(q_flat, lut_q.reshape(-1, head_dim))
    q_roped = q_roped.reshape(seq_len, emb_dim)

    k_flat = k.reshape(seq_len, n_kv_heads, head_dim).reshape(
        seq_len * n_kv_heads, head_dim
    )
    k_roped = _rope_ref(k_flat, lut_k.reshape(-1, head_dim))
    k_roped = k_roped.reshape(seq_len, kv_dim)

    # 7. FlashAttention
    attn_out = attention_reference(
        q_roped, k_roped, v, n_heads, n_kv_heads, head_dim, causal=True
    )

    # 8. O projection
    proj = (attn_out.astype(np.float32) @ wo.astype(np.float32)).astype(bfloat16)

    # 9. Residual add
    res1 = (proj.astype(np.float32) + x_in.astype(np.float32)).astype(bfloat16)

    # 10. FFN RMSNorm
    normed2 = _rms_norm_ref(res1, ffn_norm_w, eps)

    # 11-12. Gate/Up GEMMs
    normed2_f32 = normed2.astype(np.float32)
    gate = (normed2_f32 @ w_gate.astype(np.float32)).astype(bfloat16)
    up = (normed2_f32 @ w_up.astype(np.float32)).astype(bfloat16)

    # 13. SwiGLU: SiLU(gate) * up
    gate_f32 = gate.astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-gate_f32))
    swiglu = (gate_f32 * sigmoid * up.astype(np.float32)).astype(bfloat16)

    # 14. Down projection
    down = (swiglu.astype(np.float32) @ w_down.astype(np.float32)).astype(bfloat16)

    # 15. FFN residual add
    output = (down.astype(np.float32) + res1.astype(np.float32)).astype(bfloat16)

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEQ_LEN = 2048
    EMB_DIM = 2048
    KV_DIM = 512
    HIDDEN_DIM = 8192
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(
        description="Full Transformer Block -- 15-launch multi-launch test"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print combined MLIR and exit",
    )
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

    print(
        f"Transformer Block Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, "
        f"kv={KV_DIM}, hidden={HIDDEN_DIM}, heads={N_HEADS}/{N_KV_HEADS}, "
        f"dk={HEAD_DIM}"
    )

    module = build_transformer_block_module(
        seq_len=SEQ_LEN,
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        hidden_dim=HIDDEN_DIM,
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
            instance_name="transformer_block",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run: build test data, run, verify ----
    np.random.seed(42)

    # Inputs (weights and activations)
    x_in = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    attn_norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    wq = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    wk = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)
    wv = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)
    wo = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    ffn_norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    w_gate = np.random.uniform(-0.1, 0.1, (EMB_DIM, HIDDEN_DIM)).astype(bfloat16)
    w_up = np.random.uniform(-0.1, 0.1, (EMB_DIM, HIDDEN_DIM)).astype(bfloat16)
    w_down = np.random.uniform(-0.01, 0.01, (HIDDEN_DIM, EMB_DIM)).astype(bfloat16)

    # RoPE LUTs (seq-first: repeated per head)
    from rope_lut.rope_lut import generate_lut

    base_lut = generate_lut(SEQ_LEN, HEAD_DIM, bfloat16)  # (SEQ_LEN, HEAD_DIM)
    lut_q = np.repeat(base_lut, N_HEADS, axis=0)  # (N_HEADS*SEQ_LEN, HEAD_DIM)
    lut_k = np.repeat(base_lut, N_KV_HEADS, axis=0)  # (N_KV_HEADS*SEQ_LEN, HEAD_DIM)

    # Intermediate buffers (zeroed)
    normed_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    q_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    v_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    q_roped_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_roped_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    attn_out_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    proj_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    res1_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    normed2_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    gate_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    up_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    swiglu_buf = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=bfloat16)
    down_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)

    # Output buffer (1D)
    output_buf = np.zeros(SEQ_LEN * EMB_DIM, dtype=bfloat16)

    # CPU reference
    print("Computing CPU reference...")
    output_ref = transformer_block_reference(
        x_in,
        attn_norm_w,
        wq,
        wk,
        wv,
        lut_q,
        lut_k,
        wo,
        ffn_norm_w,
        w_gate,
        w_up,
        w_down,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
    )

    # Func signature: 27 args
    # inputs = args 0-25 (including zeroed intermediates)
    # expected_outputs = arg 26 (1D output)
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="transformer_block",
    )

    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,  # arg0:  x_in
                attn_norm_w,  # arg1:  attn_norm_w
                normed_buf,  # arg2:  normed (intermediate)
                wq,  # arg3:  wq
                q_buf,  # arg4:  q (intermediate)
                wk,  # arg5:  wk
                k_buf,  # arg6:  k (intermediate)
                wv,  # arg7:  wv
                v_buf,  # arg8:  v (intermediate)
                lut_q.flatten(),  # arg9:  lut_q (1D)
                lut_k.flatten(),  # arg10: lut_k (1D)
                q_roped_buf,  # arg11: q_roped (intermediate)
                k_roped_buf,  # arg12: k_roped (intermediate)
                attn_out_buf,  # arg13: attn_out (intermediate)
                wo,  # arg14: wo
                proj_buf,  # arg15: proj (intermediate)
                res1_buf,  # arg16: res1 (intermediate)
                ffn_norm_w,  # arg17: ffn_norm_w
                normed2_buf,  # arg18: normed2 (intermediate)
                w_gate,  # arg19: w_gate
                gate_buf,  # arg20: gate (intermediate)
                w_up,  # arg21: w_up
                up_buf,  # arg22: up (intermediate)
                swiglu_buf,  # arg23: swiglu (intermediate)
                w_down,  # arg24: w_down
                down_buf,  # arg25: down (intermediate)
            ],
            expected_outputs=[
                output_ref.reshape(-1),  # arg26: output (1D)
            ],
            rtol=0.2,
            atol=4.0,
            min_correlation=0.95,
        )
    )
