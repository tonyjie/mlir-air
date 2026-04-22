# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K4: gemm (64, 2048) x (2048, 8192) -> (64, 8192) BF16 standalone.

Llama-3.2-1B chunked-prefill kernel — Gate / Up FFN projections at C=64.
Mirrors K2/K3; N=8192 (4x larger than K2's N=2048). With tile_n=128 the
N dim takes 16 N-iters per herd_n=4 column.
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llama3.kernel_builder.gemm_builder import _build_gemm_module

from air.backend.xrt_runner import XRTRunner

from _profile import profile_kernel

M = 64  # chunk size
K = 2048  # emb_dim
N = 8192  # hidden_dim

TILE_M = 64
TILE_K_L2 = 64
TILE_K_L1 = 32
TILE_N = 128
HERD_M = 1
HERD_N = 4


def _build():
    return _build_gemm_module(
        m=M,
        k=K,
        n=N,
        tile_m=TILE_M,
        tile_k_l2=TILE_K_L2,
        tile_k_l1=TILE_K_L1,
        tile_n=TILE_N,
        herd_m=HERD_M,
        herd_n=HERD_N,
    )


def run_correctness():
    print(
        f"K4 gemm: ({M},{K}) x ({K},{N}) -> ({M},{N}) bf16, "
        f"tile_m={TILE_M} tile_n={TILE_N} herd={HERD_M}x{HERD_N}"
    )
    np.random.seed(0)
    a = (np.random.randn(M, K) * 0.1).astype(bfloat16)
    b = (np.random.randn(K, N) * 0.1).astype(bfloat16)
    c_expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K4_gemm_64x2048_to_8192",
    )
    return runner.run_test(
        _build(),
        inputs=[a, b],
        expected_outputs=[c_expected],
        # 524288 elements -> one BF16 boundary noise tip; atol=3.0 absorbs it.
        rtol=5e-2,
        atol=3.0,
    )


def run_profile(iterations=20, warmup=5):
    np.random.seed(0)
    a = (np.random.randn(M, K) * 0.1).astype(bfloat16)
    b = (np.random.randn(K, N) * 0.1).astype(bfloat16)
    out_buf = np.zeros((M, N), dtype=bfloat16)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)
    return profile_kernel(
        _build,
        inputs=[a, b, out_buf],
        gflops_per_invocation=2.0 * M * K * N / 1e9,
        herd_active=HERD_M * HERD_N,
        instance_name="K4_gemm_64x2048_to_8192_profile",
        label=f"K4 gemm ({M},{K})x({K},{N}) tile_m={TILE_M} herd={HERD_M}x{HERD_N}",
        iterations=iterations,
        warmup=warmup,
        expected_output=expected,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    sys.exit(
        run_profile(args.iterations, args.warmup) if args.profile else run_correctness()
    )


if __name__ == "__main__":
    main()
