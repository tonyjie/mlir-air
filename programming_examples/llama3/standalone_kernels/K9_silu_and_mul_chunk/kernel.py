# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K9: silu_and_mul chunk standalone (gate, up shape (64, 8192) bf16).

Llama-3.2-1B chunked-prefill kernel — SwiGLU activation:
  output[i] = SiLU(gate[i]) * up[i]   where SiLU(x) = x * sigmoid(x)

Production silu_and_mul kernel is 1D-flat. We view the (chunk_size,
hidden_dim) = (64, 8192) input as flat n=524288.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))

from _llm_shared.kernel_builder.ffn_swiglu.silu_and_mul import build_module
from _llm_shared.kernel_builder.external_kernels import compile_silu_and_mul

from air.backend.xrt_runner import XRTRunner

CHUNK = 64
HIDDEN_DIM = 8192
N = CHUNK * HIDDEN_DIM  # 524288, total flat element count
TILE_N = 1024  # production default tile size


def silu_reference(x):
    """SiLU(x) = x * sigmoid(x), in F32."""
    xf = x.astype(np.float32)
    return xf / (1.0 + np.exp(-xf))


def main():
    print(
        f"K9 silu_and_mul: chunk={CHUNK} x hidden_dim={HIDDEN_DIM} "
        f"-> flat n={N}, tile_n={TILE_N}"
    )

    print("  Compiling silu_and_mul.cc -> silu_and_mul.o ...")
    compile_silu_and_mul()

    mlir_module = build_module(N, TILE_N, bfloat16)

    np.random.seed(2)
    gate = np.random.uniform(-4.0, 4.0, N).astype(bfloat16)
    up = np.random.uniform(-4.0, 4.0, N).astype(bfloat16)
    expected = (silu_reference(gate) * up.astype(np.float32)).astype(bfloat16)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K9_silu_and_mul_chunk",
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[gate, up],
        expected_outputs=[expected],
        rtol=5e-2,
        atol=5e-1,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
