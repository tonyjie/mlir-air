# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K10: eltwise_add chunk standalone (a, b shape (64, 2048) bf16).

Llama-3.2-1B chunked-prefill kernel — residual add (used twice in o_ffn_chunk):
  output[i] = a[i] + b[i]

Production eltwise_add kernel is 1D-flat; we view (chunk, emb_dim) =
(64, 2048) as flat n=131072.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))

from eltwise_add.eltwise_add import build_module

from air.backend.xrt_runner import XRTRunner

CHUNK = 64
EMB_DIM = 2048
N = CHUNK * EMB_DIM  # 131072
TILE_N = 2048  # divides N (131072 / (2048 * herd_y=2) = 32 outer iters)
VECTOR_SIZE = 16
HERD_Y = 2  # default num_tiles in eltwise_add.build_module


def main():
    print(
        f"K10 eltwise_add: chunk={CHUNK} x emb_dim={EMB_DIM} "
        f"-> flat n={N}, tile_n={TILE_N}, herd_y={HERD_Y}"
    )

    mlir_module = build_module(
        N, TILE_N, bfloat16, vector_size=VECTOR_SIZE, num_tiles=HERD_Y
    )

    np.random.seed(3)
    a = np.random.randn(N).astype(bfloat16)
    b = np.random.randn(N).astype(bfloat16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K10_eltwise_add_chunk",
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[a, b],
        expected_outputs=[expected],
        rtol=1e-2,
        atol=5e-2,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
