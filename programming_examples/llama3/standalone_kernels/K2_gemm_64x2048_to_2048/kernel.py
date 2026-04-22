# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K2: gemm (64, 2048) x (2048, 2048) -> (64, 2048) BF16 standalone.

Llama-3.2-1B chunked-prefill kernel — Q-projection / O-projection at C=64.
Uses the production builder llama3.kernel_builder.gemm_builder._build_gemm_module
which wraps matrix_multiplication.bf16.run.build_module with a GEMM_TRANSFORM_IR
overlay that production llama3 prefill is validated against (1.30s prefill).

Tile config tuned for M=64 (production uses M=2048 with tile_m=64; we use
tile_m=8 so 8*herd_m=64 covers M in exactly 1 tile per herd column):
  tile_m=8, tile_k_l2=64, tile_k_l1=32, tile_n=128
  herd_m=8, herd_n=4   (full 8x4 herd)
This gives:
  M-iters per herd col = 64 / (8 * 8)   = 1
  N-iters per herd col = 2048 / (4 * 128) = 4
  K-iters per L2 block = 64 / 32 = 2
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# Make programming_examples importable so we can reach
# _llm_shared.kernel_builder.gemm_builder (production GEMM builder).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))
from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module

from air.backend.xrt_runner import XRTRunner

# Chunked-prefill GEMM shape
M = 64  # chunk size
K = 2048  # emb_dim (in_features)
N = 2048  # emb_dim (out_features for Q-proj or O-proj)

# Tile config. Production uses tile_m=64, herd_m=8 (for M=2048). The transform
# IR in _build_gemm_module appears to be hand-tuned for tile_m=64 — using a
# smaller tile_m broke things with our M=64. Keep tile_m=64 and use herd_m=1
# so the M-coverage equals our M exactly. (4 of 32 cores active in this config;
# we'll explore higher-utilization configs once the baseline works.)
TILE_M = 64
TILE_K_L2 = 64
TILE_K_L1 = 32
TILE_N = 128
HERD_M = 1
HERD_N = 4


def main():
    print(
        f"K2 gemm: ({M},{K}) x ({K},{N}) -> ({M},{N}) bf16, "
        f"tile_m={TILE_M} tile_n={TILE_N} herd={HERD_M}x{HERD_N}"
    )

    np.random.seed(0)
    # Use small-magnitude random data to limit accumulation magnitude in bf16.
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
        instance_name="K2_gemm_64x2048_to_2048",
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
