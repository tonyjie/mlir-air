#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 standalone NPU validation for Qwen3-4B's truly-new shapes.

Most leaf-kernel × shape combinations Qwen3-4B needs are already validated
by sibling deployments (see kernel_registry/qwen3_4b.md "carry-over" rows).
This script runs cold standalone NPU tests on the shapes that are NEW to
the catalog:

  GEMM 2048 × 2560 × 9728   (Gate/Up: K=emb_dim_2560, N=hidden_dim_9728)
  GEMM 2048 × 9728 × 2560   (Down:    K=hidden_dim_9728, N=emb_dim_2560)
  GEMV M=9728  K=2560        (Gate/Up decode: K=emb_2560, M=hidden_9728)
  GEMV M=2560  K=2560        (RMS+O decode at emb_2560)
  RMSNorm M=2048 N=2560      (per-block input + final norm)

Each test compiles + runs on real NPU2 with cosine vs CPU F32 reference.
PASS gate per supported_kernels.md: cosine >= 0.999 (GEMM/RMSNorm),
XRTRunner default for GEMV.
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
sys.path.insert(0, _EXAMPLES)
sys.path.insert(0, os.path.join(_EXAMPLES, "_llm_shared"))
sys.path.insert(0, os.path.join(_EXAMPLES, "matrix_vector_multiplication", "bf16"))
sys.path.insert(0, os.path.join(_EXAMPLES, "rms_norm"))

from air.backend.xrt_runner import XRTRunner

from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module


def cosine(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_gemm(name, m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n):
    print(f"\n{'='*60}\n  GEMM {name}: M={m}, K={k}, N={n}")
    print(
        f"  tile=({tile_m},{tile_k_l2},{tile_k_l1},{tile_n}) herd=({herd_m},{herd_n})"
    )
    print(f"{'='*60}")

    module = _build_gemm_module(
        m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n
    )

    np.random.seed(42)
    a = (np.random.randn(m, k) * 1.0).astype(bfloat16)
    b = (np.random.randn(k, n) * 0.1).astype(bfloat16)
    c_ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="matmul_bf16",
        runtime_loop_tiling_sizes=[2, 2],
    )
    t0 = time.time()
    rc = runner.run_test(
        module,
        inputs=[a, b],
        expected_outputs=[c_ref],
        rtol=8e-2,
        atol=2.0,
        min_correlation=0.999,
    )
    dt = time.time() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"  -> {status}  (compile+run {dt:.1f}s)")
    return rc == 0, dt


def run_gemv(name, m, k, tile_m=8, m_input=4, herd_m=8):
    """Use the matvec.py builder directly via standalone harness pattern."""
    print(f"\n{'='*60}\n  GEMV {name}: M={m}, K={k}, tile_m={tile_m} m_input={m_input}")
    print(f"{'='*60}")

    import matvec  # programming_examples/matrix_vector_multiplication/bf16/matvec.py

    module = matvec.build_module(
        m,
        k,
        tile_m,
        m_input,
        herd_m,
        bfloat16,
        bfloat16,
    )

    np.random.seed(42)
    A = (np.random.randn(m, k) * 0.1).astype(bfloat16)
    x = (np.random.randn(k) * 1.0).astype(bfloat16)
    y_ref = (A.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="matvec_bf16",
    )
    t0 = time.time()
    rc = runner.run_test(
        module,
        inputs=[A, x],
        expected_outputs=[y_ref],
        rtol=8e-2,
        atol=2.0,
        min_correlation=0.999,
    )
    dt = time.time() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"  -> {status}  (compile+run {dt:.1f}s)")
    return rc == 0, dt


def run_rmsnorm(name, M, N, herd_x=8):
    """Standalone RMSNorm test via the shared rmsnorm_multitile harness pattern."""
    print(f"\n{'='*60}\n  RMSNorm {name}: M={M}, N={N}, herd_x={herd_x}")
    print(f"{'='*60}")

    sys.path.insert(
        0, os.path.join(_EXAMPLES, "_llm_shared", "kernel_builder", "rmsnorm_multitile")
    )
    sys.path.insert(0, os.path.join(_EXAMPLES, "weighted_rms_norm"))

    from _llm_shared.kernel_builder.stitching import _wrap_ir_in_launch
    import weighted_rms_norm as wrn
    from air.ir import Context, Module

    bare_module_text = str(wrn.build_module(M, N, bfloat16, 16, herd_x=herd_x))
    wrapped_text = _wrap_ir_in_launch(bare_module_text)
    with Context() as ctx:
        from air.ir import Location

        with Location.unknown():
            wrapped = Module.parse(wrapped_text)

    np.random.seed(42)
    x = (np.random.randn(M, N) * 1.0).astype(bfloat16)
    w = (np.random.randn(N) * 1.0).astype(bfloat16)
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
    y_ref = ((x_f32 / rms) * w.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="weighted_rms_norm",
    )
    t0 = time.time()
    rc = runner.run_test(
        wrapped,
        inputs=[x, w],
        expected_outputs=[y_ref],
        rtol=2e-1,
        atol=2.0,
        min_correlation=0.999,
    )
    dt = time.time() - t0
    status = "PASS" if rc == 0 else "FAIL"
    print(f"  -> {status}  (compile+run {dt:.1f}s)")
    return rc == 0, dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 standalone NPU tests for Qwen3-4B new shapes"
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Filter: gemm-q | gemm-kv | gemm-gateup | gemm-o | gemm-down | gemv-gateup | gemv-q | rms | all",
    )
    args = parser.parse_args()

    results = {}

    # GEMM tests — production tile configs mirror qwen25_3b for the same K/N alignments
    if args.only in ("all", "gemm-q"):
        # Q proj: K=emb=2560, N=q_dim=4096
        results["gemm_q"] = run_gemm(
            "Q",
            2048,
            2560,
            4096,
            tile_m=64,
            tile_k_l2=256,
            tile_k_l1=32,
            tile_n=64,
            herd_m=8,
            herd_n=4,
        )
    if args.only in ("all", "gemm-kv"):
        # K/V proj: K=emb=2560, N=kv_dim=1024
        results["gemm_kv"] = run_gemm(
            "K/V",
            2048,
            2560,
            1024,
            tile_m=64,
            tile_k_l2=64,
            tile_k_l1=32,
            tile_n=128,
            herd_m=8,
            herd_n=4,
        )
    if args.only in ("all", "gemm-gateup"):
        # Gate/Up: K=emb=2560, N=hidden=9728
        results["gemm_gateup"] = run_gemm(
            "Gate/Up",
            2048,
            2560,
            9728,
            tile_m=64,
            tile_k_l2=64,
            tile_k_l1=32,
            tile_n=64,
            herd_m=8,
            herd_n=4,
        )
    if args.only in ("all", "gemm-o"):
        # O proj: K=q_dim=4096, N=emb=2560
        results["gemm_o"] = run_gemm(
            "O",
            2048,
            4096,
            2560,
            tile_m=64,
            tile_k_l2=256,
            tile_k_l1=32,
            tile_n=64,
            herd_m=8,
            herd_n=4,
        )
    if args.only in ("all", "gemm-down"):
        # Down: K=hidden=9728, N=emb=2560
        results["gemm_down"] = run_gemm(
            "Down",
            2048,
            9728,
            2560,
            tile_m=64,
            tile_k_l2=256,
            tile_k_l1=32,
            tile_n=64,
            herd_m=8,
            herd_n=4,
        )

    # GEMV tests — production tile config from qwen25_3b decode path
    if args.only in ("all", "gemv-gateup"):
        # Gate/Up decode: M=hidden=9728, K=emb=2560
        results["gemv_gateup"] = run_gemv("Gate/Up", 9728, 2560)
    if args.only in ("all", "gemv-q"):
        # Q decode: M=q_dim=4096, K=emb=2560
        results["gemv_q"] = run_gemv("Q", 4096, 2560)

    # RMSNorm at emb=2560
    if args.only in ("all", "rms"):
        results["rmsnorm"] = run_rmsnorm("attn_norm", 2048, 2560, herd_x=8)

    print(f"\n\n{'='*60}\n  PHASE 1 SUMMARY\n{'='*60}")
    n_pass = sum(1 for v in results.values() if v[0])
    n_total = len(results)
    for name, (ok, dt) in results.items():
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name:20s}  ({dt:.1f}s)")
    print(f"\n  {n_pass}/{n_total} PASS")
    sys.exit(0 if n_pass == n_total else 1)
