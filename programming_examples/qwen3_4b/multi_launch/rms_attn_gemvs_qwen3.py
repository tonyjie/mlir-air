# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rms_attn_gemvs_qwen3 — Step 2 of the bottom-up Qwen3 NPU decode build.

Produces a 4-launch decode ELF that does RMSNorm + Q/K/V GEMV WITHOUT RoPE.
The omitted RoPE step is Qwen3's architectural requirement: per-head Q/K
RMSNorm (Q/K Norm) MUST happen between Q/K projection and RoPE, and RMSNorm
doesn't commute with RoPE under asymmetric weights — so the standard fused
`rms_gemv_rope` ELF can't be used.

Uses ONLY leaf kernels validated standalone in `qwen3_kernel_registry_test.py`:
  - `_build_rms_1d` (from llama3 rms_gemv_rope_multi) — RMSNorm M=1 with 1D I/O
  - `matvec.build_module` — GEMV M=q_dim/kv_dim, K=emb_dim

q_dim parameter handles Qwen3's q_dim != emb_dim case (16 heads × 128 head_dim
= 2048 ≠ emb_dim=1024).

Func signature (9 args, 4 launches; inputs first, outputs last for
XRTRunner compatibility):
    %arg0: x_in        memref<emb_dim xbf16>            RMSNorm input
    %arg1: norm_w      memref<emb_dim xbf16>            RMSNorm weight
    %arg2: wq          memref<q_dim  x emb_dim xbf16>
    %arg3: wk          memref<kv_dim x emb_dim xbf16>
    %arg4: wv          memref<kv_dim x emb_dim xbf16>
    %arg5: normed      memref<emb_dim xbf16>            RMSNorm output / GEMV B
    %arg6: q_out       memref<q_dim xbf16>
    %arg7: k_out       memref<kv_dim xbf16>
    %arg8: v_out       memref<kv_dim xbf16>
"""

import os
import re
import sys

from ml_dtypes import bfloat16

# Ensure llama3 builder helpers + matvec are importable.
_THIS_DIR = os.path.dirname(__file__)
_EXAMPLES = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (
    _EXAMPLES,
    os.path.join(_EXAMPLES, "llama3"),
    os.path.join(_EXAMPLES, "matrix_vector_multiplication", "bf16"),
    # NOTE: do NOT add weighted_rms_norm dir directly — it would shadow
    # the namespace-package import used by llama3 helpers
    # (`from weighted_rms_norm.weighted_rms_norm import build_module`).
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _rename_all_for_gemv_or_rms(text, prefix):
    """SSA renaming that preserves matvec + rmsnorm extern kernel symbols."""
    extern = {
        "@matvec_vectorized_bf16_bf16",
        "@linalg_fill_bf16",
    }
    # Affine maps
    for n in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(n) + r"(?!\w)", f"#{prefix}_{n[1:]}", text)
    # Named SSA
    for n in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(n) + r"(?!\w)", f"%{prefix}_{n[1:]}", text)
    # Numbered SSA
    for n in sorted(
        set(re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = text.replace(n, f"%{prefix}_n{n[1:]}")
    # Symbol names except externs
    for n in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if n not in extern:
            text = text.replace(n, f"@{prefix}_{n[1:]}")
    return text


def build_rms_attn_gemvs_qwen3_module(
    emb_dim=1024,
    q_dim=2048,
    kv_dim=1024,
    tile_m=8,
    m_input=4,
    herd_m=8,
):
    """Build the 4-launch RMSNorm + Q/K/V GEMV decode ELF for Qwen3."""
    from llama3.multi_launch_builder.rms_gemv_rope_multi import _build_rms_1d
    from llama3.multi_launch_builder.superseded.ffn_full_multi import (
        _extract_between_func_and_return,
        _fix_launch_func_args,
    )
    from matvec import build_module as build_gemv

    # ---- Build sub-kernels ----
    print(f"  [1/4] RMSNorm 1D (M=1, N={emb_dim})...")
    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, vector_size=16))

    print(f"  [2/4] Q GEMV (m={q_dim}, k={emb_dim})...")
    q_ir = str(build_gemv(q_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [3/4] K GEMV (m={kv_dim}, k={emb_dim})...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print(f"  [4/4] V GEMV (m={kv_dim}, k={emb_dim})...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    # ---- Extract maps ----
    def _extract_affine_maps(ir_text):
        return [l for l in ir_text.split("\n") if l.startswith("#")]

    # ---- Stitch ----
    # Combined arg layout:
    #   0 x_in, 1 norm_w, 2 normed, 3 wq, 4 q_out, 5 wk, 6 k_out, 7 wv, 8 v_out
    #
    # Per sub-kernel arg remap (sub-kernel arg → combined arg):
    #   RMSNorm 1D rms_norm_1d(x_1d, weight, out_1d):  {0:0, 1:1, 2:2}
    #   GEMV    matmul(A[m,k], B[k], C[m]):             {0:wA, 1:2, 2:wC}
    # Combined arg layout (inputs first, outputs last for XRTRunner compat):
    #   0 x_in, 1 norm_w, 2 wq, 3 wk, 4 wv, 5 normed, 6 q_out, 7 k_out, 8 v_out
    # Sub-kernel arg → combined arg:
    #   RMSNorm 1D (x_1d, weight, out_1d): {0:0, 1:1, 2:5}
    #   GEMV (A, B=normed, C):             {0:wA, 1:5, 2:Cout}
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "rms", {0: 0, 1: 1, 2: 5}),
        (q_ir, "q", {0: 2, 1: 5, 2: 6}),
        (k_ir, "k", {0: 3, 1: 5, 2: 7}),
        (v_ir, "v", {0: 4, 1: 5, 2: 8}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all_for_gemv_or_rms(body, prefix)
        maps = [_rename_all_for_gemv_or_rms(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # ---- Private func decls (deduplicated across all sub-kernels) ----
    privates = set()
    for ir in (rms_ir, q_ir, k_ir, v_ir):
        for line in ir.split("\n"):
            if "func.func private" in line:
                privates.add(line.strip())
    privates_str = "\n  ".join(sorted(privates))

    # ---- Assemble combined module ----
    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @rms_attn_gemvs(
    %arg0: memref<{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{q_dim}x{emb_dim}xbf16>,
    %arg3: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg4: memref<{kv_dim}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<{q_dim}xbf16>,
    %arg7: memref<{kv_dim}xbf16>,
    %arg8: memref<{kv_dim}xbf16>
  ) {{
{chr(10).join(bodies)}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
    print(f"  Module: {len(combined.splitlines())} lines, 9 args, 4 launches")
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
    parser.add_argument("-p", "--print-module-only", action="store_true")
    args = parser.parse_args()

    print(
        f"Qwen3 rms_attn_gemvs_qwen3 standalone test "
        f"(emb_dim={args.emb_dim}, q_dim={args.q_dim}, kv_dim={args.kv_dim})"
    )

    module = build_rms_attn_gemvs_qwen3_module(
        emb_dim=args.emb_dim,
        q_dim=args.q_dim,
        kv_dim=args.kv_dim,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Random inputs at realistic deployment magnitudes (weights ~ 0.02, x ~ 1).
    np.random.seed(42)
    eps = 1e-6
    x_in = (np.random.randn(args.emb_dim) * 1.0).astype(bfloat16)
    norm_w = (np.random.randn(args.emb_dim) * 0.1 + 1.0).astype(bfloat16)
    wq = (np.random.randn(args.q_dim, args.emb_dim) * 0.02).astype(bfloat16)
    wk = (np.random.randn(args.kv_dim, args.emb_dim) * 0.02).astype(bfloat16)
    wv = (np.random.randn(args.kv_dim, args.emb_dim) * 0.02).astype(bfloat16)

    # CPU reference
    x_f32 = x_in.astype(np.float32)
    w_f32 = norm_w.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32) + eps)
    normed_ref = ((x_f32 / rms) * w_f32).astype(bfloat16)
    q_ref = (wq.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)
    k_ref = (wk.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)
    v_ref = (wv.astype(np.float32) @ normed_ref.astype(np.float32)).astype(bfloat16)

    # Use the backend directly so we can read back NPU outputs and report
    # cosine_sim per output rather than rely on XRTRunner's element-wise check
    # (which is too noise-tight for a stitched 4-launch BF16 pipeline at K=1024).
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
        instance_name="rms_attn_gemvs",
    )
    t0 = time.time()
    artifact = backend.compile(module)
    print(f"  Compile: {time.time()-t0:.1f}s")

    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)

    # Fresh output buffers (zeroed)
    normed_buf = np.zeros(args.emb_dim, dtype=bfloat16)
    q_buf = np.zeros(args.q_dim, dtype=bfloat16)
    k_buf = np.zeros(args.kv_dim, dtype=bfloat16)
    v_buf = np.zeros(args.kv_dim, dtype=bfloat16)
    inputs_all = [x_in, norm_w, wq, wk, wv, normed_buf, q_buf, k_buf, v_buf]
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

    for idx in (5, 6, 7, 8):
        bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _read(idx, shape):
        return bos[idx].read(sizes[idx], 0).view(np.int16).view(bfloat16).reshape(shape)

    npu_normed = _read(5, (args.emb_dim,))
    npu_q = _read(6, (args.q_dim,))
    npu_k = _read(7, (args.kv_dim,))
    npu_v = _read(8, (args.kv_dim,))
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
        "v": _cos(npu_v, v_ref),
    }
    for name, c in cs.items():
        marker = "PASS" if c > 0.99 else "FAIL"
        print(f"  [{marker}] {name:7s} cosine={c:.6f}")
    sys.exit(0 if all(c > 0.99 for c in cs.values()) else 1)
