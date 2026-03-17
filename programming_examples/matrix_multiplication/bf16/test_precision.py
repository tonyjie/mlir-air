#!/usr/bin/env python3
"""Measure AIR GEMM BF16 precision against CPU F32 reference.

Uses IRON's input range (A=randn*4, B=rand*4) for fair comparison.
Compiles via run.py's transform IR (correct for BF16 output).

Usage:
    cd programming_examples/matrix_multiplication/bf16
    python3 test_precision.py --m 2048 --k 2048 --n 2048 \
        --tile-m 64 --tile-k-l2 256 --tile-k-l1 32 --tile-n 64 \
        --herd-m 8 --herd-n 4
"""

import argparse, sys, os, re, shutil
import numpy as np
from ml_dtypes import bfloat16
import filelock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import build_module
from air.ir import Module
from air.dialects.air import run_transform
from air.backend.xrt import XRTBackend


def load_transform_ir():
    with open(os.path.join(os.path.dirname(__file__), "run.py")) as f:
        match = re.search(r'transform_ir_string = """(.+?)"""', f.read(), re.DOTALL)
    return match.group(1)


def clean_build():
    for d in ["build_peano/air_project", "air_project"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-k-l2", type=int, default=256)
    parser.add_argument("--tile-k-l1", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--herd-m", type=int, default=8)
    parser.add_argument("--herd-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    M, K, N = args.m, args.k, args.n
    transform_str = load_transform_ir()

    # Use PyTorch to generate inputs matching IRON exactly
    import torch

    torch.manual_seed(args.seed)
    A_torch = torch.randn(M, K, dtype=torch.bfloat16) * 4
    B_torch = torch.rand(K, N, dtype=torch.bfloat16) * 4

    # Convert to numpy BF16 for AIR
    A = A_torch.view(torch.int16).numpy().view(bfloat16)
    B = B_torch.view(torch.int16).numpy().view(bfloat16)
    C = np.zeros((M, N), dtype=bfloat16)
    ref_f32 = A.astype(np.float32) @ B.astype(np.float32)

    print(
        f"GEMM {M}x{K}x{N}, herd={args.herd_m}x{args.herd_n}, "
        f"tile={args.tile_m}x{args.tile_k_l2}x{args.tile_k_l1}x{args.tile_n}"
    )
    print(
        f"Input: A=randn*4 [{float(A.min()):.1f},{float(A.max()):.1f}], "
        f"B=rand*4 [{float(B.min()):.1f},{float(B.max()):.1f}]"
    )
    print(f"Output range: [{ref_f32.min():.0f}, {ref_f32.max():.0f}]")

    # Compile
    clean_build()
    os.chdir("build_peano")
    mlir = build_module(
        M,
        K,
        N,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
        bfloat16,
        bfloat16,
        arch="aie2p",
        direct_codegen=True,
    )
    tir = Module.parse(
        "module attributes {transform.with_named_sequence} {" + transform_str + "}",
        context=mlir.context,
    )
    run_transform(tir, mlir)

    backend = XRTBackend(omit_while_true_loop=False, runtime_loop_tiling_sizes=[2, 2])
    artifact = backend.compile(mlir)

    # Run on NPU
    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)
        # Warmup
        for _ in range(3):
            invoker(A, B, C)
        # Measured run
        results = invoker(A, B, C)

    npu_f32 = results[-1].reshape(M, N).astype(np.float32)
    backend.unload()
    os.chdir("..")

    # Precision metrics
    diff = np.abs(npu_f32 - ref_f32)
    corr = np.corrcoef(npu_f32.flatten(), ref_f32.flatten())[0, 1]
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"\nPrecision (vs CPU F32 reference):")
    print(f"  corr:     {corr:.8f}")
    print(f"  max_err:  {max_err:.2f}")
    print(f"  mean_err: {mean_err:.4f}")
    print(
        f"  max_err as % of range: {max_err / (ref_f32.max() - ref_f32.min()) * 100:.2f}%"
    )

    # Error percentiles
    pcts = [50, 90, 95, 99, 99.9, 100]
    print(f"\n  Error percentiles:")
    for p in pcts:
        print(f"    {p:5.1f}%: {np.percentile(diff, p):.4f}")

    # IRON-style tolerance check: diff < max(abs_tol, rel_tol * (|a| + |b|))
    npu_abs = np.abs(npu_f32)
    ref_abs = np.abs(ref_f32)
    total = M * N
    for rel_tol in [0.04, 0.10, 0.50]:
        iron_tol = np.maximum(1e-6, rel_tol * (npu_abs + ref_abs))
        fails = np.sum(diff > iron_tol)
        status = "PASS" if fails == 0 else "FAIL"
        print(
            f"\n  IRON tolerance (rel_tol={rel_tol}): {fails}/{total} fails ({100*fails/total:.2f}%) [{status}]"
        )
