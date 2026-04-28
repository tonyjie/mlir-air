# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase B Step 2 — verify the existing 8-launch o_gemv_ffn ELF compiles +
runs at Qwen3 decode shapes.

Reuses `llama3.multi_launch_builder.o_gemv_ffn_multi.build_o_gemv_ffn_module`
unchanged — it was extended with `o_in_dim` kwarg during Phase 1 prefill.
For Qwen3 decode all four GEMVs (O, Gate, Up, Down) use tile_m=8, sharing
the same `mv.o` extern (DIM_M_OUTPUT=8). No symbol rename needed.

Func args (15 total, mirrors llama3 production):
    arg0  attn_out (o_in_dim,)        intermediate input from attention
    arg1  wo       (emb_dim, o_in_dim)
    arg2  proj     (emb_dim,)         intermediate
    arg3  x_res    (emb_dim,)         residual input (= block input)
    arg4  res1     (emb_dim,)         intermediate (proj + x_res)
    arg5  ffn_n_w  (emb_dim,)         FFN RMSNorm weight
    arg6  normed2  (emb_dim,)         intermediate
    arg7  w_gate   (hidden_dim, emb_dim)
    arg8  gate     (hidden_dim,)      intermediate
    arg9  w_up     (hidden_dim, emb_dim)
    arg10 up       (hidden_dim,)      intermediate
    arg11 swiglu   (hidden_dim,)      intermediate (SiLU(gate)*up)
    arg12 w_down   (emb_dim, hidden_dim)
    arg13 down     (emb_dim,)         intermediate
    arg14 out      (emb_dim,)         OUTPUT (down + res1)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_DEPLOY_DIR = _THIS_DIR.parent
_EXAMPLES = _DEPLOY_DIR.parent
for _p in (
    str(_EXAMPLES),
    str(_EXAMPLES / "llama3"),
    str(_EXAMPLES / "matrix_vector_multiplication" / "bf16"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    EMB_DIM = 1024
    HIDDEN_DIM = 3072
    O_IN_DIM = 2048  # n_heads * head_dim for Qwen3

    print(
        f"Phase B Step 2: o_gemv_ffn ELF at Qwen3 decode shapes "
        f"(emb_dim={EMB_DIM}, hidden_dim={HIDDEN_DIM}, o_in_dim={O_IN_DIM})"
    )

    from llama3.multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module
    from _llm_shared.kernel_builder.external_kernels import (
        compile_all_external_kernels,
    )

    print("\nCompiling external kernels (mv.o, silu_and_mul.o, ...)")
    compile_all_external_kernels(head_dim=128)

    print("\nBuilding module (8 launches)...")
    # tile_m=8 for ALL four GEMVs (same mv.o extern, no rename needed):
    #   O    : m=emb_dim=1024,  k=o_in_dim=2048
    #   Gate : m=hidden_dim=3072, k=emb_dim=1024
    #   Up   : m=hidden_dim=3072, k=emb_dim=1024
    #   Down : m=emb_dim=1024, k=hidden_dim=3072
    # All M values are multiples of 64 (= tile_m * herd_m), all K values are
    # multiples of 64 (vector width).
    module = build_o_gemv_ffn_module(
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        o_in_dim=O_IN_DIM,
        tile_m=8,
        m_input=4,
        herd_m=8,
        down_tile_m=8,  # match Qwen3 — no shape collision
        down_m_input=4,
        down_k_split=None,  # K=3072 fits standard auto-split (32*255=8160 ≥ 3072)
    )

    from air.backend.xrt import XRTBackend
    import filelock
    import pyxrt as xrt

    print("\nCompiling (might take ~30s)...")
    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
        runtime_loop_tiling_sizes=[4, 4],
        use_lock_race_condition_fix=True,
        output_format="elf",
        instance_name="o_gemv_ffn",
    )
    t0 = time.time()
    try:
        artifact = backend.compile(module)
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        sys.exit(2)
    print(f"  Compile: {time.time()-t0:.1f}s")

    # ---- Random inputs ----
    np.random.seed(13)
    attn_out_in = (np.random.randn(O_IN_DIM) * 1.0).astype(bfloat16)
    wo = (np.random.randn(EMB_DIM, O_IN_DIM) * 0.02).astype(bfloat16)
    x_res_in = (np.random.randn(EMB_DIM) * 1.0).astype(bfloat16)
    ffn_norm_w = (np.random.randn(EMB_DIM) * 0.1 + 1.0).astype(bfloat16)
    w_gate = (np.random.randn(HIDDEN_DIM, EMB_DIM) * 0.02).astype(bfloat16)
    w_up = (np.random.randn(HIDDEN_DIM, EMB_DIM) * 0.02).astype(bfloat16)
    w_down = (np.random.randn(EMB_DIM, HIDDEN_DIM) * 0.02).astype(bfloat16)

    # ---- CPU reference ----
    eps = 1e-6
    attn_f32 = attn_out_in.astype(np.float32)
    wo_f32 = wo.astype(np.float32)
    proj_ref = wo_f32 @ attn_f32  # (emb_dim,)
    x_f32 = x_res_in.astype(np.float32)
    res1_ref = (proj_ref + x_f32).astype(bfloat16)

    res1_f32 = res1_ref.astype(np.float32)
    rms = np.sqrt(np.mean(res1_f32 * res1_f32) + eps)
    normed2_ref = ((res1_f32 / rms) * ffn_norm_w.astype(np.float32)).astype(bfloat16)

    n2_f32 = normed2_ref.astype(np.float32)
    gate_ref = (w_gate.astype(np.float32) @ n2_f32).astype(bfloat16)
    up_ref = (w_up.astype(np.float32) @ n2_f32).astype(bfloat16)

    g_f32 = gate_ref.astype(np.float32)
    u_f32 = up_ref.astype(np.float32)
    silu = g_f32 * (1.0 / (1.0 + np.exp(-g_f32)))
    swiglu_ref = (silu * u_f32).astype(bfloat16)

    down_ref = (w_down.astype(np.float32) @ swiglu_ref.astype(np.float32)).astype(
        bfloat16
    )
    out_ref = (down_ref.astype(np.float32) + res1_ref.astype(np.float32)).astype(
        bfloat16
    )

    # ---- Run on NPU ----
    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)

    proj_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    res1_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    normed2_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    gate_buf = np.zeros(HIDDEN_DIM, dtype=bfloat16)
    up_buf = np.zeros(HIDDEN_DIM, dtype=bfloat16)
    swiglu_buf = np.zeros(HIDDEN_DIM, dtype=bfloat16)
    down_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    out_buf = np.zeros(EMB_DIM, dtype=bfloat16)

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

    # Read back outputs
    for idx in (2, 4, 6, 8, 10, 11, 13, 14):
        bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _read(idx, n):
        return bos[idx].read(sizes[idx], 0).view(np.int16).view(bfloat16).reshape(n)

    npu_proj = _read(2, EMB_DIM)
    npu_res1 = _read(4, EMB_DIM)
    npu_normed2 = _read(6, EMB_DIM)
    npu_gate = _read(8, HIDDEN_DIM)
    npu_up = _read(10, HIDDEN_DIM)
    npu_swiglu = _read(11, HIDDEN_DIM)
    npu_down = _read(13, EMB_DIM)
    npu_out = _read(14, EMB_DIM)
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


if __name__ == "__main__":
    main()
