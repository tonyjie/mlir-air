# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase B Step 2 — fused 8-launch o_gemv_ffn ELF for Qwen3 decode.

Qwen3-specific because of the **3-K matvec collision**:
    O GEMV    : K = q_dim       = 2048
    Gate/Up   : K = emb_dim     = 1024
    Down GEMV : K = hidden_dim  = 3072

llama3's `o_gemv_ffn_multi.build_o_gemv_ffn_module` only handles 2 K values
(K=2048 and K=8192) via a single rename (`dg_matvec_*` for Down).
For 3 distinct K shapes we need a 2nd rename: `og_matvec_*` for O.

External objects (compiled via `_llm_shared/kernel_builder/external_kernels.py`):
  - mv.o            : default symbol — used by Gate, Up (K=1024)
  - mv_og.o         : `@og_matvec_*` rename — used by O (K=2048)
  - mv_dg_qwen3.o   : `@dg_matvec_*` rename — used by Down (K=3072)

All three .o files are compiled with DIM_M_OUTPUT=8 (matches Qwen3's
uniform tile_m=8 across all four GEMVs) — only the exported symbol differs.

Func signature mirrors llama3's o_gemv_ffn (15 args, 8 launches):
    arg0  attn_out (q_dim,)        input from attention output
    arg1  wo       (emb_dim, q_dim)
    arg2  proj     (emb_dim,)      intermediate
    arg3  x_res    (emb_dim,)      residual input
    arg4  res1     (emb_dim,)      intermediate
    arg5  ffn_n_w  (emb_dim,)      FFN RMSNorm weight
    arg6  normed2  (emb_dim,)      intermediate
    arg7  w_gate   (hidden_dim, emb_dim)
    arg8  gate     (hidden_dim,)   intermediate
    arg9  w_up     (hidden_dim, emb_dim)
    arg10 up       (hidden_dim,)   intermediate
    arg11 swiglu   (hidden_dim,)   intermediate
    arg12 w_down   (emb_dim, hidden_dim)
    arg13 down     (emb_dim,)      intermediate
    arg14 out      (emb_dim,)      OUTPUT
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


def build_o_gemv_ffn_silu_qwen3_module(
    emb_dim=1024,
    hidden_dim=3072,
    o_in_dim=2048,
    tile_m=8,
    m_input=4,
    herd_m=8,
):
    """8-launch fused o+ffn ELF for Qwen3 decode (with 3-K matvec rename)."""
    from llama3.multi_launch_builder.superseded.ffn_full_multi import (
        _extract_between_func_and_return,
        _fix_launch_func_args,
        _wrap_ir_in_launch,
    )
    from _llm_shared.kernel_builder.stitching import (
        _rename_all_with_externs,
        _extract_private_funcs,
    )
    from matvec import build_module as build_gemv
    from eltwise_add.eltwise_add import build_module as build_add
    from _llm_shared.kernel_builder.ffn_swiglu.silu_and_mul import (
        build_module as build_silu,
    )

    print(f"  [1/8] O GEMV (m={emb_dim}, k={o_in_dim})...")
    o_ir = str(
        build_gemv(emb_dim, o_in_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print(f"  [2/8] Eltwise Add (post-attn residual, n={emb_dim})...")
    add1_ir = _wrap_ir_in_launch(
        str(
            build_add(
                emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1
            )
        )
    )

    # 1D RMSNorm at M=1
    print(f"  [3/8] RMSNorm 1D (M=1, N={emb_dim})...")
    from llama3.multi_launch_builder.rms_gemv_rope_multi import _build_rms_1d

    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, vector_size=16))

    print(f"  [4/8] Gate GEMV (m={hidden_dim}, k={emb_dim})...")
    gate_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print(f"  [5/8] Up GEMV (m={hidden_dim}, k={emb_dim})...")
    up_ir = str(
        build_gemv(hidden_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print(f"  [6/8] SiLU x mul (n={hidden_dim})...")
    silu_ir = _wrap_ir_in_launch(
        str(build_silu(hidden_dim, hidden_dim // 8, bfloat16, herd_x=8, herd_y=1))
    )

    print(f"  [7/8] Down GEMV (m={emb_dim}, k={hidden_dim})...")
    down_ir = str(
        build_gemv(emb_dim, hidden_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print(f"  [8/8] Eltwise Add (FFN residual, n={emb_dim})...")
    add2_ir = _wrap_ir_in_launch(
        str(
            build_add(
                emb_dim, emb_dim // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1
            )
        )
    )

    # ---- Stitch ----
    # Combined arg layout (matches llama3 o_gemv_ffn):
    #   0 attn_out, 1 wo, 2 proj, 3 x_res, 4 res1, 5 ffn_n_w, 6 normed2,
    #   7 wgate, 8 gate, 9 wup, 10 up, 11 swiglu, 12 wdown, 13 down, 14 out
    stitch_specs = [
        (o_ir, "og", {0: 1, 1: 0, 2: 2}),  # wo, attn_out → proj   (O GEMV: og_rename)
        (add1_ir, "a1", {0: 2, 1: 3, 2: 4}),  # proj, x_res → res1
        (rms_ir, "rm", {0: 4, 1: 5, 2: 6}),  # res1, ffn_n_w → normed2
        (gate_ir, "gg", {0: 7, 1: 6, 2: 8}),  # wgate, normed2 → gate (default mv.o)
        (up_ir, "ug", {0: 9, 1: 6, 2: 10}),  # wup, normed2 → up     (default mv.o)
        (silu_ir, "sw", {0: 8, 1: 10, 2: 11}),  # gate, up → swiglu
        (
            down_ir,
            "dg",
            {0: 12, 1: 11, 2: 13},
        ),  # wdown, swiglu → down  (Down GEMV: dg_rename)
        (add2_ir, "a2", {0: 13, 1: 4, 2: 14}),  # down, res1 → out
    ]

    # Externs preserved (NOT prefixed) for each launch group:
    _EXTERN_DEFAULT = {
        "@matvec_vectorized_bf16_bf16",
        "@linalg_fill_bf16",
        "@silu_and_mul_bf16",
    }
    # O GEMV (og prefix): matvec/linalg_fill REPLACED by og_*. silu preserved.
    _EXTERN_O = {"@silu_and_mul_bf16"}
    # Down GEMV (dg prefix): matvec/linalg_fill REPLACED by dg_*. silu preserved.
    _EXTERN_DOWN = {"@silu_and_mul_bf16"}

    bodies, maps_all = [], []
    for ir, prefix, arg_map in stitch_specs:
        body = _extract_between_func_and_return(ir)
        # Re-fetch maps from THIS sub-IR
        maps = [l for l in ir.split("\n") if l.startswith("#")]
        if prefix == "og":
            externs = _EXTERN_O
        elif prefix == "dg":
            externs = _EXTERN_DOWN
        else:
            externs = _EXTERN_DEFAULT
        body = _rename_all_with_externs(body, prefix, externs)
        maps = [_rename_all_with_externs(m, prefix, externs) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        # Update link_with for renamed groups
        if prefix == "og":
            body = body.replace('link_with = "mv.o"', 'link_with = "mv_og.o"')
        elif prefix == "dg":
            body = body.replace('link_with = "mv.o"', 'link_with = "mv_dg_qwen3.o"')
        bodies.append(body)
        maps_all.extend(maps)

    # ---- Private func decls (deduplicated; renamed per group; correct link_with) ----
    default_privates = _extract_private_funcs(gate_ir)
    silu_privates = _extract_private_funcs(silu_ir)

    o_privates = _extract_private_funcs(o_ir)
    o_privates_renamed = []
    for p in o_privates:
        p_renamed = _rename_all_with_externs(p, "og", _EXTERN_O)
        p_renamed = p_renamed.replace('link_with = "mv.o"', 'link_with = "mv_og.o"')
        o_privates_renamed.append(p_renamed.strip())

    down_privates = _extract_private_funcs(down_ir)
    down_privates_renamed = []
    for p in down_privates:
        p_renamed = _rename_all_with_externs(p, "dg", _EXTERN_DOWN)
        p_renamed = p_renamed.replace(
            'link_with = "mv.o"', 'link_with = "mv_dg_qwen3.o"'
        )
        down_privates_renamed.append(p_renamed.strip())

    seen_funcs = set()
    all_privates = []
    for p in (
        default_privates + o_privates_renamed + down_privates_renamed + silu_privates
    ):
        fname = re.search(r"@(\w+)", p)
        if fname and fname.group(1) not in seen_funcs:
            seen_funcs.add(fname.group(1))
            all_privates.append(p.strip())

    n_total = emb_dim  # output is 1D shape (emb_dim,) NOT n_total — let me check llama3
    # Looking at llama3's add2 output: 1D (n_total,) where n_total=seq_len*emb_dim.
    # For decode (M=1) seq_len=1 so n_total=emb_dim. We use 1D (emb_dim,).
    combined = "\n".join(maps_all) + f"""
module {{
  {chr(10).join('  ' + p for p in all_privates)}
  func.func @o_gemv_ffn_silu(
    %arg0: memref<{o_in_dim}xbf16>,
    %arg1: memref<{emb_dim}x{o_in_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{emb_dim}xbf16>,
    %arg7: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg8: memref<{hidden_dim}xbf16>,
    %arg9: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg10: memref<{hidden_dim}xbf16>,
    %arg11: memref<{hidden_dim}xbf16>,
    %arg12: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg13: memref<{emb_dim}xbf16>,
    %arg14: memref<{emb_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
    print(f"  Module: {len(combined.splitlines())} lines, 15 args, 8 launches")
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
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--o-in-dim", type=int, default=2048)
    parser.add_argument("-p", "--print-module-only", action="store_true")
    args = parser.parse_args()

    print(
        f"Phase B Step 2: o_gemv_ffn_silu_qwen3 standalone "
        f"(emb_dim={args.emb_dim}, hidden_dim={args.hidden_dim}, o_in_dim={args.o_in_dim})"
    )

    from _llm_shared.kernel_builder.external_kernels import (
        compile_all_external_kernels,
        compile_mv_og,
        compile_mv_dg_qwen3,
    )

    print(
        "\nCompiling external kernels (mv.o + silu_and_mul.o + new mv_og.o + mv_dg_qwen3.o)..."
    )
    compile_all_external_kernels(head_dim=128)
    compile_mv_og(tile_m=8)
    compile_mv_dg_qwen3(tile_m=8)

    module = build_o_gemv_ffn_silu_qwen3_module(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        o_in_dim=args.o_in_dim,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    from air.backend.xrt import XRTBackend
    import filelock
    import pyxrt as xrt

    print("\nCompiling ELF (might take ~30s)...")
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
        runtime_loop_tiling_sizes=[4, 4],
        use_lock_race_condition_fix=True,
        output_format="elf",
        instance_name="o_gemv_ffn_silu",
    )
    t0 = time.time()
    artifact = backend.compile(module)
    print(f"  Compile: {time.time()-t0:.1f}s")

    # ---- Random inputs ----
    np.random.seed(13)
    eps = 1e-6
    attn_out_in = (np.random.randn(args.o_in_dim) * 1.0).astype(bfloat16)
    wo = (np.random.randn(args.emb_dim, args.o_in_dim) * 0.02).astype(bfloat16)
    x_res_in = (np.random.randn(args.emb_dim) * 1.0).astype(bfloat16)
    ffn_norm_w = (np.random.randn(args.emb_dim) * 0.1 + 1.0).astype(bfloat16)
    w_gate = (np.random.randn(args.hidden_dim, args.emb_dim) * 0.02).astype(bfloat16)
    w_up = (np.random.randn(args.hidden_dim, args.emb_dim) * 0.02).astype(bfloat16)
    w_down = (np.random.randn(args.emb_dim, args.hidden_dim) * 0.02).astype(bfloat16)

    # ---- CPU reference ----
    proj_ref = (wo.astype(np.float32) @ attn_out_in.astype(np.float32)).astype(bfloat16)
    res1_ref = (proj_ref.astype(np.float32) + x_res_in.astype(np.float32)).astype(
        bfloat16
    )
    r_f32 = res1_ref.astype(np.float32)
    rms = np.sqrt(np.mean(r_f32 * r_f32) + eps)
    normed2_ref = ((r_f32 / rms) * ffn_norm_w.astype(np.float32)).astype(bfloat16)
    n2 = normed2_ref.astype(np.float32)
    gate_ref = (w_gate.astype(np.float32) @ n2).astype(bfloat16)
    up_ref = (w_up.astype(np.float32) @ n2).astype(bfloat16)
    g = gate_ref.astype(np.float32)
    u = up_ref.astype(np.float32)
    silu = g * (1.0 / (1.0 + np.exp(-g)))
    swiglu_ref = (silu * u).astype(bfloat16)
    down_ref = (w_down.astype(np.float32) @ swiglu_ref.astype(np.float32)).astype(
        bfloat16
    )
    out_ref = (down_ref.astype(np.float32) + res1_ref.astype(np.float32)).astype(
        bfloat16
    )

    # ---- Run on NPU ----
    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)

    proj_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    res1_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    normed2_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    gate_buf = np.zeros(args.hidden_dim, dtype=bfloat16)
    up_buf = np.zeros(args.hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(args.hidden_dim, dtype=bfloat16)
    down_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    out_buf = np.zeros(args.emb_dim, dtype=bfloat16)

    inputs_all = [
        attn_out_in,
        wo,
        proj_buf,
        x_res_in,
        res1_buf,
        ffn_norm_w,
        normed2_buf,
        w_gate,
        gate_buf,
        w_up,
        up_buf,
        swiglu_buf,
        w_down,
        down_buf,
        out_buf,
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
    print(f"\n  NPU run: {(time.time()-t1)*1000:.2f} ms")

    for idx in (2, 4, 6, 8, 10, 11, 13, 14):
        bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _read(idx, n):
        return bos[idx].read(sizes[idx], 0).view(np.int16).view(bfloat16).reshape(n)

    npu_proj = _read(2, args.emb_dim)
    npu_res1 = _read(4, args.emb_dim)
    npu_normed2 = _read(6, args.emb_dim)
    npu_gate = _read(8, args.hidden_dim)
    npu_up = _read(10, args.hidden_dim)
    npu_swiglu = _read(11, args.hidden_dim)
    npu_down = _read(13, args.emb_dim)
    npu_out = _read(14, args.emb_dim)
    backend.unload()

    def _cos(a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    print("\nCorrectness vs CPU reference:")
    cs = {
        "proj": _cos(npu_proj, proj_ref),
        "res1": _cos(npu_res1, res1_ref),
        "normed2": _cos(npu_normed2, normed2_ref),
        "gate": _cos(npu_gate, gate_ref),
        "up": _cos(npu_up, up_ref),
        "swiglu": _cos(npu_swiglu, swiglu_ref),
        "down": _cos(npu_down, down_ref),
        "out": _cos(npu_out, out_ref),
    }
    for name, c in cs.items():
        marker = "PASS" if c > 0.99 else "FAIL"
        print(f"  [{marker}] {name:8s} cosine={c:.6f}")
    sys.exit(0 if all(c > 0.99 for c in cs.values()) else 1)
