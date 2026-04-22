# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K1: rmsnorm_chunk (M=64, N=2048) BF16 standalone validation.

Llama-3.2-1B chunked-prefill kernel — RMSNorm of one chunk's hidden state.
Reuses the build_module from programming_examples/weighted_rms_norm/ unchanged
(it is fully parameterized over M, N).

Single-tile (herd_x=1) for standalone correctness only. The production
chunked multi-launch ELF will invoke this same builder with the herd_x value
that matches production rms_gemms_rope_multi.py.

Shapes:
  x      (64, 2048) bf16
  weight    (2048,) bf16
  output (64, 2048) bf16
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# Import the source builder from programming_examples/weighted_rms_norm/.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))
from weighted_rms_norm.weighted_rms_norm import build_module, rms_norm_reference

from air.backend.xrt_runner import XRTRunner

# Chunked-prefill shape constants
M = 64  # chunk size
N = 2048  # emb_dim
VECTOR_SIZE = 16
HERD_X = 1  # standalone single-tile; multi-launch ELF uses herd_x=8


def main():
    print(f"K1 rmsnorm_chunk: M={M}, N={N}, herd_x={HERD_X}")

    np.random.seed(0)
    x = np.random.rand(M, N).astype(bfloat16)
    w = np.random.rand(N).astype(bfloat16)
    y_expected = rms_norm_reference(x, w)

    mlir_module = build_module(M, N, bfloat16, VECTOR_SIZE, herd_x=HERD_X)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K1_rmsnorm_chunk",
        runtime_loop_tiling_sizes=[4, 4],
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[x, w],
        expected_outputs=[y_expected],
        rtol=5e-2,
        atol=5e-1,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
