# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K3: gemm (64, 2048) x (2048, 512) -> (64, 512) BF16 standalone.

Llama-3.2-1B chunked-prefill kernel — K-projection / V-projection at C=64
(kv_dim = n_kv_heads * head_dim = 8 * 64 = 512).

Mirrors K2 with N=512 instead of N=2048. tile_n=128 still divides N=512
cleanly (4 N-tiles per herd column = 1 N-iter per herd_n=4).
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))
from llama3.kernel_builder.gemm_builder import _build_gemm_module

from air.backend.xrt_runner import XRTRunner

M = 64  # chunk size
K = 2048  # emb_dim (in_features)
N = 512  # kv_dim = n_kv_heads * head_dim

TILE_M = 64
TILE_K_L2 = 64
TILE_K_L1 = 32
TILE_N = 128
HERD_M = 1
HERD_N = 4


def main():
    print(
        f"K3 gemm: ({M},{K}) x ({K},{N}) -> ({M},{N}) bf16, "
        f"tile_m={TILE_M} tile_n={TILE_N} herd={HERD_M}x{HERD_N}"
    )

    np.random.seed(0)
    a = (np.random.randn(M, K) * 0.1).astype(bfloat16)
    b = (np.random.randn(K, N) * 0.1).astype(bfloat16)
    c_expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)

    mlir_module = _build_gemm_module(
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

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K3_gemm_64x2048_to_512",
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[a, b],
        expected_outputs=[c_expected],
        rtol=5e-2,
        atol=2.0,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
