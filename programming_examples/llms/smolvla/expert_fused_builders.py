# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fused multi-launch ELF builders for the SmolVLA action expert.

Forks of `shared/builders/rms_gemms_rope_multi.py` and
`shared/builders/o_ffn_multi.py`. They could not simply be re-parameterised,
for one reason:

    **the expert's q_dim (960) is not its emb_dim (720).**

Both shared builders assume a square attention block -- `rms_gemms_rope_multi`
declares `wq` as `memref<emb_dim x emb_dim>` and `o_ffn_multi` looks up
`gemm_registry_config(seq_len, emb_dim, emb_dim)` for the O projection. That
holds for every model ported so far (the SmolVLA *backbone* is 960/960) but not
for the expert, whose hidden state is 720 wide while its 15 query heads make
q_dim = 15*64 = 960. Same bug family as the MHA scratch-arg issue: an
assumption about attention geometry baked into a shape.

Three builders, matching the expert's two layer kinds:

    build_expert_rms_qkv_rope_module   EVEN layers (self-attn):  6 launches
        RMSNorm + Q/K/V GEMM + RoPE Q + RoPE K
    build_expert_rms_q_rope_module     ODD layers (cross-attn):  3 launches
        RMSNorm + Q GEMM + RoPE Q
        (k/v come from the 241-token backbone prefix, projected once per
         inference outside the denoise loop -- not per layer, not per step)
    build_expert_o_ffn_module          BOTH layer kinds:         8 launches
        O proj + residual + RMSNorm + gate + up + SiLU-and-Mul + down + residual

Two hazards this file handles explicitly, both of which have already cost a
debugging round elsewhere in the project:

1.  **tile_m is not the method default.** `gemm_registry_config` asserts the
    registry tile_m equals the method's (drain=32 / fused-cast=64). Every expert
    shape is M=64, which forces tile_m=16, so that assert would fire. We call the
    raw `gemm_config` lookup instead and override tile_m from the registry -- the
    same thing `vision_prefill.py` does by hand for the connector.

2.  **The symbol suffix must key on tile_m AND tile_n.** `compile_gemm_mm` bakes
    both DIM_M and DIM_N into `mm.o` at compile time, but the shared helpers
    (`disambiguate_by_tile_n`, `vit_fused_builders._force_tile_n_suffix`)
    hardcode the tag as "m32"/"m64" from the *method*, ignoring the actual
    tile_m. For the expert that would name a tile_m=16 object `mm_m32_n80.o`,
    which collides with the *backbone's* genuine tile_m=32 / tile_n=80 object
    (`256x960x320`) -- two different DIM_M values, one filename, silent garbage.
    `_expert_gemm_spec` below keys the suffix on `_m{tile_m}_n{tile_n}`.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# `..`    -> programming_examples/llms   (shared.*)
# `../..` -> programming_examples        (kernel_registry, weighted_rms_norm, silu_and_mul)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.ir import *
from air.dialects.air import *
from air.backend.xrt_runner import type_mapper

from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
    alloc_gemm_scratch,
)
from shared.builders.rms_gemms_rope_multi import _build_rope_2d
from shared.builders.o_ffn_multi import _build_add_2d_to_2d

# ---------------------------------------------------------------------------
# Expert config (read off the live lerobot/smolvla_base checkpoint)
# ---------------------------------------------------------------------------
EXPERT_EMB_DIM = 720
EXPERT_Q_DIM = 960  # 15 heads x 64 -- NOT emb_dim
EXPERT_KV_DIM = 320  # 5 heads x 64
EXPERT_HIDDEN_DIM = 2048
EXPERT_N_HEADS = 15
EXPERT_N_KV_HEADS = 5
EXPERT_HEAD_DIM = 64
EXPERT_SEQ = 64  # 50 action tokens padded to 64 (tile_m=16 x herd_m=4)
EXPERT_PREFIX = 241  # backbone K/V cache length


# ---------------------------------------------------------------------------
# Registry lookup that tolerates the expert's tile_m
# ---------------------------------------------------------------------------
def _expert_gemm_spec(m, k, n):
    """Registry-driven build recipe for one expert GEMM.

    Like `gemm_registry_config`, but:
      - takes tile_m from the registry rather than from the method default
        (every expert shape is M=64 -> tile_m=16, which the shared helper's
        assert rejects), and
      - keys sym_suffix / mm.o on BOTH tile_m and tile_n, so a tile_m=16 object
        can never collide with a tile_m=32 object of the same tile_n.

    Also surfaces the registry's per-shape `herd` override, which the expert
    needs on nearly every row (M=64 forces herd_m=4; N=720 forces herd_n=3).
    """
    from kernel_registry.registry_lookup import gemm_config
    from shared.builders.gemm_builder import gemm_method_spec

    cfg = gemm_config(m, k, n, "bf16", "high")
    tile = cfg["tile"]
    spec = dict(gemm_method_spec(cfg["method"]))
    spec["method"] = cfg["method"]
    spec["tile_m"] = tile["tile_m"]  # registry wins over the method default
    spec["tile_k_l2"] = tile["tile_k_l2"]
    spec["tile_k_l1"] = tile["tile_k_l1"]
    spec["tile_n"] = tile["tile_n"]
    spec["herd"] = tuple(cfg.get("herd", (8, 4)))
    spec["shape"] = (m, k, n)

    # Key the symbol suffix on ALL THREE dims baked into mm.o by
    # compile_gemm_mm (external_kernels.py:158-160: DIM_M=tile_m, DIM_N=tile_n,
    # DIM_K=tile_k_l1). The shared helpers key on tile_n alone, which is a
    # latent hazard the expert actually trips: q/k/v resolve to
    # (tile_m=16, tile_k_l1=48, tile_n=80) while o/down resolve to
    # (16, 32, 80) -- same method, same tile_n, DIFFERENT DIM_K. Under the
    # tile_n-only naming both want "mm_m16_n80.o", so whichever ELF compiles
    # last overwrites the other's object and that ELF links a wrong-DIM_K
    # microkernel: no error, wrong numbers.
    sfx = f"_m{tile['tile_m']}_k{tile['tile_k_l1']}_n{tile['tile_n']}"
    obj = f"mm{sfx}.o"
    spec["sym_suffix"] = sfx
    spec["obj"] = obj
    spec["build_kwargs"] = dict(spec["build_kwargs"])
    spec["build_kwargs"]["sym_suffix"] = sfx
    spec["build_kwargs"]["link_with_name"] = obj

    # The launch grid must iterate exactly once in M: the external-mm.o emit
    # path returns garbage (mean_rel_L1 ~0.67, no error) when it does not.
    # See smolvla_backbone_prefill.py:383-397.
    herd_m = spec["herd"][0]
    assert m % (tile["tile_m"] * herd_m) == 0, (
        f"{m}x{k}x{n}: M={m} not divisible by tile_m*herd_m="
        f"{tile['tile_m']}*{herd_m}"
    )
    assert m // tile["tile_m"] // herd_m == 1, (
        f"{m}x{k}x{n}: M-direction launch grid would iterate "
        f"{m // tile['tile_m'] // herd_m} times; the external-mm.o path is only "
        f"correct at 1 (smolvla_backbone_prefill.py:383)"
    )
    assert tile["tile_m"] % 16 == 0, f"tile_m={tile['tile_m']} violates mm_aie2p.cc"
    return spec


def _gemm_ir(spec):
    """Build one GEMM sub-module from an _expert_gemm_spec."""
    from shared.builders.gemm_builder import _build_gemm_module

    m, k, n = spec["shape"]
    herd_m, herd_n = spec["herd"]
    return str(
        _build_gemm_module(
            m,
            k,
            n,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            herd_m,
            herd_n,
            **spec["build_kwargs"],
        )
    )


def _gemm_externs(spec):
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
    if sc is not None:  # fused-cast: {0:in, 1:w, 2:C-f32-scratch, 3:bf16-out}
        return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
    return {0: in_idx, 1: w_idx, 2: out_idx}  # drain: {0:in, 1:w, 2:bf16-out}


def expert_gemm_specs(
    seq_len=EXPERT_SEQ,
    emb_dim=EXPERT_EMB_DIM,
    q_dim=EXPERT_Q_DIM,
    kv_dim=EXPERT_KV_DIM,
    hidden_dim=EXPERT_HIDDEN_DIM,
    prefix_len_padded=256,
):
    """Every GEMM the expert port compiles, so the driver can stage the right
    `mm.o` objects before building any ELF. Keyed by a stable name."""
    return {
        "q": _expert_gemm_spec(seq_len, emb_dim, q_dim),
        "kv": _expert_gemm_spec(seq_len, emb_dim, kv_dim),
        "o": _expert_gemm_spec(seq_len, q_dim, emb_dim),
        "gate_up": _expert_gemm_spec(seq_len, emb_dim, hidden_dim),
        "down": _expert_gemm_spec(seq_len, hidden_dim, emb_dim),
        "kv_cross": _expert_gemm_spec(prefix_len_padded, kv_dim, kv_dim),
    }


# ---------------------------------------------------------------------------
# EVEN layers: RMSNorm + Q/K/V GEMM + RoPE Q + RoPE K  (6 launches, 13 args)
# ---------------------------------------------------------------------------
def build_expert_rms_qkv_rope_module(
    seq_len=EXPERT_SEQ,
    emb_dim=EXPERT_EMB_DIM,
    q_dim=EXPERT_Q_DIM,
    kv_dim=EXPERT_KV_DIM,
    n_heads=EXPERT_N_HEADS,
    n_kv_heads=EXPERT_N_KV_HEADS,
    head_dim=EXPERT_HEAD_DIM,
    rope_herd_x=8,
    print_kernels=False,
):
    """EVEN (self-attn) expert layer front half.

    Identical in structure to `build_rms_gemms_rope_module`, but Q is
    (emb_dim -> q_dim) instead of (emb_dim -> emb_dim).

    Func @expert_rms_qkv_rope, 13 memref args:
        %arg0  x_in     (seq_len, emb_dim)
        %arg1  norm_w   (emb_dim,)
        %arg2  normed   (seq_len, emb_dim)
        %arg3  wq       (emb_dim, q_dim)     <-- not square
        %arg4  q        (seq_len, q_dim)
        %arg5  wk       (emb_dim, kv_dim)
        %arg6  k        (seq_len, kv_dim)
        %arg7  wv       (emb_dim, kv_dim)
        %arg8  v        (seq_len, kv_dim)
        %arg9  lut_q    (seq_len*q_dim,)
        %arg10 lut_k    (seq_len*kv_dim,)
        %arg11 q_roped  (seq_len, q_dim)
        %arg12 k_roped  (seq_len, kv_dim)
    """
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    assert q_dim == n_heads * head_dim, (q_dim, n_heads, head_dim)
    assert kv_dim == n_kv_heads * head_dim, (kv_dim, n_kv_heads, head_dim)

    q_spec = _expert_gemm_spec(seq_len, emb_dim, q_dim)
    k_spec = _expert_gemm_spec(seq_len, emb_dim, kv_dim)
    v_spec = k_spec  # same shape

    q_total = seq_len * q_dim
    k_total = seq_len * kv_dim

    print("  [ex_qkv 1/6] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(
        f"  [ex_qkv 2/6] Q GEMM {seq_len}x{emb_dim}x{q_dim} "
        f"({q_spec['method']} tm={q_spec['tile_m']} tn={q_spec['tile_n']} "
        f"herd={q_spec['herd']})..."
    )
    q_ir = _gemm_ir(q_spec)
    print(
        f"  [ex_qkv 3-4/6] K/V GEMM {seq_len}x{emb_dim}x{kv_dim} "
        f"({k_spec['method']} tm={k_spec['tile_m']} tn={k_spec['tile_n']} "
        f"herd={k_spec['herd']})..."
    )
    k_ir = _gemm_ir(k_spec)
    v_ir = k_ir
    print(f"  [ex_qkv 5/6] RoPE Q (outer={seq_len}x{q_dim}, dk={head_dim})...")
    rope_q_ir = str(_build_rope_2d(seq_len, q_dim, head_dim, bfloat16, rope_herd_x))
    print(f"  [ex_qkv 6/6] RoPE K (outer={seq_len}x{kv_dim}, dk={head_dim})...")
    rope_k_ir = str(_build_rope_2d(seq_len, kv_dim, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMM", q_ir),
            ("K GEMM", k_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'=' * 60}\n  Sub-kernel: {name}\n{'=' * 60}\n{ir}")

    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (q_spec, seq_len, q_dim),
            (k_spec, seq_len, kv_dim),
            (v_spec, seq_len, kv_dim),
        ],
        base_arg_count=13,
    )

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{q_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{q_total}xbf16>"),
        FuncArg("%arg10", f"memref<{k_total}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms=_gemm_externs(q_spec),
        ),
        KernelSlice(
            k_ir,
            "k",
            _gemm_arg_map(2, 5, 6, scratch_for[1]),
            extern_syms=_gemm_externs(k_spec),
        ),
        KernelSlice(
            v_ir,
            "v",
            _gemm_arg_map(2, 7, 8, scratch_for[2]),
            extern_syms=_gemm_externs(v_spec),
            private_from=False,  # same suffix as the K slice; do not re-declare
        ),
        KernelSlice(rope_q_ir, "rq", {0: 4, 1: 9, 2: 11}, extern_syms={"@rope"}),
        KernelSlice(
            rope_k_ir,
            "rk",
            {0: 6, 1: 10, 2: 12},
            extern_syms={"@rope"},
            private_from=False,
        ),
    ]

    module = stitch_elf(
        "expert_rms_qkv_rope", base_args, slices, scratch_args=scratch_args
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ---------------------------------------------------------------------------
# ODD layers: RMSNorm + Q GEMM + RoPE Q  (3 launches, 7 args)
# ---------------------------------------------------------------------------
def build_expert_rms_q_rope_module(
    seq_len=EXPERT_SEQ,
    emb_dim=EXPERT_EMB_DIM,
    q_dim=EXPERT_Q_DIM,
    n_heads=EXPERT_N_HEADS,
    head_dim=EXPERT_HEAD_DIM,
    rope_herd_x=8,
    print_kernels=False,
):
    """ODD (cross-attn) expert layer front half.

    Cross layers project only Q from the action tokens -- their K and V come
    from re-projecting the backbone's constant 241-token prefix cache, which
    this port hoists out of the 10-step denoise loop entirely (it does not
    depend on x_t, so lerobot recomputing it 10 times is pure waste).

    Func @expert_rms_q_rope, 7 memref args:
        %arg0  x_in     (seq_len, emb_dim)
        %arg1  norm_w   (emb_dim,)
        %arg2  normed   (seq_len, emb_dim)
        %arg3  wq       (emb_dim, q_dim)
        %arg4  q        (seq_len, q_dim)
        %arg5  lut_q    (seq_len*q_dim,)
        %arg6  q_roped  (seq_len, q_dim)
    """
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    assert q_dim == n_heads * head_dim, (q_dim, n_heads, head_dim)
    q_spec = _expert_gemm_spec(seq_len, emb_dim, q_dim)
    q_total = seq_len * q_dim

    print("  [ex_q 1/3] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(
        f"  [ex_q 2/3] Q GEMM {seq_len}x{emb_dim}x{q_dim} "
        f"({q_spec['method']} tm={q_spec['tile_m']} tn={q_spec['tile_n']})..."
    )
    q_ir = _gemm_ir(q_spec)
    print(f"  [ex_q 3/3] RoPE Q (outer={seq_len}x{q_dim}, dk={head_dim})...")
    rope_q_ir = str(_build_rope_2d(seq_len, q_dim, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [("RMSNorm", rms_ir), ("Q GEMM", q_ir), ("RoPE Q", rope_q_ir)]:
            print(f"\n{'=' * 60}\n  Sub-kernel: {name}\n{'=' * 60}\n{ir}")

    scratch_args, scratch_for = alloc_gemm_scratch(
        [(q_spec, seq_len, q_dim)], base_arg_count=7
    )

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{q_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{q_total}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{q_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms=_gemm_externs(q_spec),
        ),
        KernelSlice(rope_q_ir, "rq", {0: 4, 1: 5, 2: 6}, extern_syms={"@rope"}),
    ]

    module = stitch_elf(
        "expert_rms_q_rope", base_args, slices, scratch_args=scratch_args
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ---------------------------------------------------------------------------
# BOTH layer kinds: O proj + residual + RMSNorm + FFN  (8 launches, 15 args)
# ---------------------------------------------------------------------------
def build_expert_o_ffn_module(
    seq_len=EXPERT_SEQ,
    emb_dim=EXPERT_EMB_DIM,
    q_dim=EXPERT_Q_DIM,
    hidden_dim=EXPERT_HIDDEN_DIM,
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    print_kernels=False,
):
    """Expert layer back half, shared by both layer kinds.

    Identical in structure to `build_o_ffn_module`, but the O projection is
    (q_dim -> emb_dim) = 960 -> 720 rather than square, so args 0 and 1 carry
    q_dim where the shared builder carries emb_dim.

    Func @expert_o_ffn, 15 memref args:
        %arg0  attn_out   (seq_len, q_dim)      <-- q_dim, not emb_dim
        %arg1  wo         (q_dim, emb_dim)      <-- not square
        %arg2  o          (seq_len, emb_dim)
        %arg3  residual   (seq_len, emb_dim)    the block input
        %arg4  x1         (seq_len, emb_dim)    o + residual
        %arg5  ffn_norm_w (emb_dim,)
        %arg6  normed2    (seq_len, emb_dim)
        %arg7  w_gate     (emb_dim, hidden_dim)
        %arg8  gate       (seq_len, hidden_dim)
        %arg9  w_up       (emb_dim, hidden_dim)
        %arg10 up         (seq_len, hidden_dim)
        %arg11 swiglu     (seq_len, hidden_dim)
        %arg12 w_down     (hidden_dim, emb_dim)
        %arg13 down       (seq_len, emb_dim)
        %arg14 out        (seq_len, emb_dim)    = down + x1, feeds the next block
    """
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from silu_and_mul.silu_and_mul import build_module_2d as build_swiglu

    o_spec = _expert_gemm_spec(seq_len, q_dim, emb_dim)
    g_spec = _expert_gemm_spec(seq_len, emb_dim, hidden_dim)
    d_spec = _expert_gemm_spec(seq_len, hidden_dim, emb_dim)
    n_total = seq_len * emb_dim

    print(
        f"  [ex_offn 1/8] O GEMM {seq_len}x{q_dim}x{emb_dim} "
        f"({o_spec['method']} tm={o_spec['tile_m']} tn={o_spec['tile_n']} "
        f"herd={o_spec['herd']})..."
    )
    o_ir = _gemm_ir(o_spec)
    print("  [ex_offn 2/8] Residual add...")
    res_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))
    print("  [ex_offn 3/8] FFN RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )
    print(
        f"  [ex_offn 4-5/8] Gate/Up GEMM {seq_len}x{emb_dim}x{hidden_dim} "
        f"({g_spec['method']} tm={g_spec['tile_m']} tn={g_spec['tile_n']} "
        f"herd={g_spec['herd']})..."
    )
    gate_ir = _gemm_ir(g_spec)
    up_ir = gate_ir
    print("  [ex_offn 6/8] SwiGLU...")
    swiglu_ir = _wrap_ir_in_launch(
        str(
            build_swiglu(
                seq_len,
                hidden_dim,
                swiglu_tile_n,
                bfloat16,
                swiglu_herd_x,
                swiglu_herd_y,
            )
        )
    )
    print(
        f"  [ex_offn 7/8] Down GEMM {seq_len}x{hidden_dim}x{emb_dim} "
        f"({d_spec['method']} tm={d_spec['tile_m']} tn={d_spec['tile_n']} "
        f"herd={d_spec['herd']})..."
    )
    down_ir = _gemm_ir(d_spec)
    # The shared o_ffn ends on a 2D->1D add (its caller wanted a flat output).
    # The expert's block output feeds straight into the next block's 2D x_in, so
    # we reuse the 2D->2D add here instead and skip a reshape on the host. It is
    # the same IR as the residual add above; the stitcher's per-slice tag keeps
    # the two instances' symbols apart.
    print("  [ex_offn 8/8] FFN add (2D -> 2D)...")
    ffn_add_ir = str(_build_add_2d_to_2d(seq_len, emb_dim, bfloat16))

    if print_kernels:
        for name, ir in [
            ("O GEMM", o_ir),
            ("Res Add", res_add_ir),
            ("FFN RMSNorm", rms_ir),
            ("Gate GEMM", gate_ir),
            ("SwiGLU", swiglu_ir),
            ("Down GEMM", down_ir),
            ("FFN Add", ffn_add_ir),
        ]:
            print(f"\n{'=' * 60}\n  Sub-kernel: {name}\n{'=' * 60}\n{ir}")

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{q_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{q_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{emb_dim}x{hidden_dim}xbf16>"),
        FuncArg("%arg10", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{hidden_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{hidden_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg13", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg14", f"memref<{seq_len}x{emb_dim}xbf16>"),
    ]
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (o_spec, seq_len, emb_dim),
            (g_spec, seq_len, hidden_dim),
            (g_spec, seq_len, hidden_dim),
            (d_spec, seq_len, emb_dim),
        ],
        base_arg_count=15,
    )

    slices = [
        KernelSlice(
            o_ir,
            "og",
            _gemm_arg_map(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(o_spec),
        ),
        KernelSlice(res_add_ir, "ra", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rms_ir, "rm", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(
            gate_ir,
            "gg",
            _gemm_arg_map(6, 7, 8, scratch_for[1]),
            extern_syms=_gemm_externs(g_spec),
        ),
        KernelSlice(
            up_ir,
            "ug",
            _gemm_arg_map(6, 9, 10, scratch_for[2]),
            extern_syms=_gemm_externs(g_spec),
            private_from=False,
        ),
        KernelSlice(
            swiglu_ir,
            "sw",
            {0: 8, 1: 10, 2: 11},
            extern_syms={"@silu_and_mul_bf16"},
        ),
        KernelSlice(
            down_ir,
            "dg",
            _gemm_arg_map(11, 12, 13, scratch_for[3]),
            extern_syms=_gemm_externs(d_spec),
        ),
        KernelSlice(ffn_add_ir, "fa", {0: 13, 1: 4, 2: 14}, private_from=False),
    ]

    module = stitch_elf("expert_o_ffn", base_args, slices, scratch_args=scratch_args)
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["even", "odd", "offn", "all"], default="all")
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--print-kernels", action="store_true")
    args = ap.parse_args()

    print("Expert GEMM specs resolved from the registry:")
    for name, s in expert_gemm_specs().items():
        m, k, n = s["shape"]
        print(
            f"  {name:9s} {m}x{k}x{n:<5} {s['method']:6s} "
            f"tiles {s['tile_m']}/{s['tile_k_l2']}/{s['tile_k_l1']}/{s['tile_n']} "
            f"herd {s['herd']}  obj {s['obj']}"
        )

    todo = ["even", "odd", "offn"] if args.which == "all" else [args.which]
    for w in todo:
        print(f"\n=== building {w} ===")
        if w == "even":
            mod = build_expert_rms_qkv_rope_module(print_kernels=args.print_kernels)
        elif w == "odd":
            mod = build_expert_rms_q_rope_module(print_kernels=args.print_kernels)
        else:
            mod = build_expert_o_ffn_module(print_kernels=args.print_kernels)
        if args.print_module_only:
            print(mod)
