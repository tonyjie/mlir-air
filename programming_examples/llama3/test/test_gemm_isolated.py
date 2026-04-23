"""Isolated precision check for llama3 prefill GEMM kernels on NPU2.

Builds each prefill GEMM shape via the same `_build_gemm_module` used in the
multi-launch ELFs, compiles via XRTBackend directly, runs on hardware, and
reports full-output precision metrics matching
`programming_examples/matrix_multiplication/bf16/test_precision.py`:
  - corr, max_err, mean_err
  - IRON-style tolerance fail %: diff > max(atol, rel * (|a|+|b|))
    at rel = 4%, 10%, 50%

Doc reference (gemm.md, 2026-03-19, mm/bf16 path):
  Q/O 2048^3:        corr=0.99992, mean_err=4.01,  8.0% fail at 4%
  K/V 2048x2048x512: corr=0.99992, mean_err=4.01,  8.0% fail at 4%
  Gate/Up 2048x2048x8192: corr=0.99992, mean_err=4.01,  8.0% fail at 4%
  Down 2048x8192x2048:    corr=0.99979, mean_err=12.53, 10.7% fail at 4%

Usage (from programming_examples/llama3):
    python3 test/test_gemm_isolated.py            # all unique shapes
    python3 test/test_gemm_isolated.py --only Q
"""

import argparse
import os
import sys
import numpy as np
import filelock
from ml_dtypes import bfloat16

# Match llama3_inference.py path setup so `llama3.kernel_builder.gemm_builder`
# and `matrix_multiplication.bf16.run` both resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from llama3.kernel_builder.gemm_builder import _build_gemm_module
from air.backend.xrt import XRTBackend

# Llama-3.2-1B prefill GEMM shapes. Per-shape tile config from gemm.md
# "Ready to Integrate" table (lines 158-162).
SHAPES = {
    "Q": dict(
        M=2048, K=2048, N=2048, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64
    ),
    "K": dict(M=2048, K=2048, N=512, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128),
    "Gate": dict(
        M=2048, K=2048, N=8192, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128
    ),
    "Down": dict(
        M=2048, K=8192, N=2048, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64
    ),
}
HERD = dict(herd_m=8, herd_n=4)


def run_one(name, M, K, N, tile_m, tile_k_l2, tile_k_l1, tile_n, verbose=False):
    print(
        f"\n{'='*70}\n"
        f"  {name} GEMM  {M}x{K}x{N}  "
        f"tile={tile_m}x{tile_k_l2}x{tile_k_l1}x{tile_n}  herd=8x4\n"
        f"{'='*70}"
    )

    # Match test_precision.py inputs (torch RNG -> reproducible vs doc).
    import torch

    torch.manual_seed(42)
    A_t = torch.randn(M, K, dtype=torch.bfloat16) * 4
    B_t = torch.rand(K, N, dtype=torch.bfloat16) * 4
    A = A_t.view(torch.int16).numpy().view(bfloat16)
    B = B_t.view(torch.int16).numpy().view(bfloat16)
    C = np.zeros((M, N), dtype=bfloat16)

    ref_f32 = A.astype(np.float32) @ B.astype(np.float32)
    print(
        f"  Input range: A=[{float(A.min()):.1f},{float(A.max()):.1f}] "
        f"B=[{float(B.min()):.1f},{float(B.max()):.1f}]  "
        f"Output range: [{ref_f32.min():.0f}, {ref_f32.max():.0f}]"
    )

    mlir_module = _build_gemm_module(
        M, K, N, tile_m, tile_k_l2, tile_k_l1, tile_n, **HERD
    )

    # gemm_builder's transform IR already does outer [2,2] tiling internally,
    # so do NOT pass runtime_loop_tiling_sizes here.
    backend = XRTBackend(verbose=verbose, omit_while_true_loop=False)
    artifact = backend.compile(mlir_module)

    with filelock.FileLock("/tmp/npu.lock"):
        invoker = backend.load(artifact)
        for _ in range(3):  # warmup
            invoker(A, B, C)
        results = invoker(A, B, C)
    npu_f32 = results[-1].reshape(M, N).astype(np.float32)
    backend.unload()

    diff = np.abs(npu_f32 - ref_f32)
    corr = float(np.corrcoef(npu_f32.flatten(), ref_f32.flatten())[0, 1])
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    print(f"\n  corr={corr:.8f}  mean_err={mean_err:.4f}  max_err={max_err:.2f}")

    npu_abs, ref_abs = np.abs(npu_f32), np.abs(ref_f32)
    total = M * N
    iron_results = {}
    for rel in [0.04, 0.10, 0.50]:
        tol = np.maximum(1e-6, rel * (npu_abs + ref_abs))
        fails = int(np.sum(diff > tol))
        pct = 100.0 * fails / total
        iron_results[rel] = pct
        print(f"  IRON tol rel={rel:.2f}: {fails:>9d}/{total} = {pct:6.2f}% fail")

    return dict(corr=corr, mean_err=mean_err, max_err=max_err, fail4=iron_results[0.04])


# Doc-recorded baseline (gemm.md, mm/bf16 standalone path).
DOC_BASELINE = {
    "Q": dict(corr=0.99992, mean_err=4.01, fail4=8.0),
    "K": dict(corr=0.99992, mean_err=4.01, fail4=8.0),
    "Gate": dict(corr=0.99992, mean_err=4.01, fail4=8.0),
    "Down": dict(corr=0.99979, mean_err=12.53, fail4=10.7),
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(SHAPES))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    shapes = {args.only: SHAPES[args.only]} if args.only else SHAPES
    metrics = {}
    for name, dims in shapes.items():
        try:
            metrics[name] = run_one(name, **dims, verbose=args.verbose)
        except Exception as e:
            print(f"!!! {name} raised: {type(e).__name__}: {e}")
            metrics[name] = None

    print(
        f"\n{'='*70}\n  Summary: llama3 path vs gemm.md baseline (mm/bf16 path)\n{'='*70}"
    )
    print(f"  {'Shape':<6} {'corr':<22} {'mean_err':<22} {'fail@4%':<22}")
    for name, m in metrics.items():
        b = DOC_BASELINE[name]
        if m is None:
            print(f"  {name:<6} ERROR")
            continue
        print(
            f"  {name:<6} "
            f"{m['corr']:.5f} (vs {b['corr']:.5f})   "
            f"{m['mean_err']:6.2f}  (vs {b['mean_err']:5.2f})    "
            f"{m['fail4']:5.2f}% (vs {b['fail4']:.1f}%)"
        )
