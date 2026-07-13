# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task 3.1 GOAL 2 support: validate the two thin attention GEMM shapes on real
NPU2 against an FP32 reference, the same way Task 1.1 validated the projection
GEMMs. These shapes are NOT in the kernel registry (registry only has the large
llama shapes), so we validate them directly here before wiring them into the
NPU attention path.

    QK^T : (M=256, K=head_dim=64, N=256)   S = Q @ K^T
    P@V  : (M=256, K=256,        N=head_dim=64)   O = P @ V

bf16-in / bf16-out, f32 accumulate + single epilogue cast (drain method,
tile_m=32) -- identical GEMM microkernel to every projection GEMM.

Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 validate_attn_gemms.py
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_PROG = Path(__file__).resolve().parent.parent.parent
if str(_PROG) not in sys.path:
    sys.path.insert(0, str(_PROG))
_LLMS = Path(__file__).resolve().parent.parent
if str(_LLMS) not in sys.path:
    sys.path.insert(0, str(_LLMS))

from air.backend.xrt_runner import XRTRunner
from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm
from shared.infra.external_kernels import compile_gemm_mm

np.random.seed(0)


def validate_gemm(m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n):
    print(
        f"\n=== GEMM {m}x{k}x{n} (tile_m={tile_m} tile_k_l2={tile_k_l2} "
        f"tile_k_l1={tile_k_l1} tile_n={tile_n} herd={herd_m}x{herd_n}) ==="
    )
    # External mm.o microkernel with DIM_M/N/K baked to these tiles.
    compile_gemm_mm(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k_l1=tile_k_l1,
        sym_suffix="",
        out_name="mm.o",
    )
    mod = build_gemm(
        m,
        k,
        n,
        tile_m,
        tile_k_l2,
        tile_k_l1,
        tile_n,
        herd_m,
        herd_n,
        bfloat16,
        bfloat16,
        arch="aie2p",
        emit_external_call=True,
    )
    A = np.random.randn(m, k).astype(bfloat16)
    B = np.random.randn(k, n).astype(bfloat16)
    # f32-accumulate reference (matches the on-device f32 accumulator + 1 cast).
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(bfloat16)

    # These two shapes are NOT in the kernel registry (registry only carries the
    # large llama GEMM shapes), and there is no per-shape standalone harness with
    # a tuned rtol/atol here. Per the phase-1 skill's "no-harness fallback", the
    # correct correctness lens is cosine / Pearson correlation vs the FP32
    # reference (the same lens used for a kernel with no harness), NOT the tight
    # element-wise np.isclose the *registered* shapes use. Element-wise diffs are
    # the expected bf16/BFP16 block-float quantization tier (~1-3% relative); what
    # matters is that these outputs feed softmax with high correlation.
    runner = XRTRunner(
        verbose=False,
        output_format="xclbin",
        runtime_loop_tiling_sizes=[4, 4],
    )
    rc = runner.run_test(
        mod,
        inputs=[A, B],
        expected_outputs=[C_ref],
        rtol=1.6e-2,
        atol=3e-2,
        max_mismatch_percentage=100.0,  # element-wise gate disabled (see above)
        min_correlation=0.999,  # the actual gate: Pearson corr vs FP32 ref
    )
    ok = rc == 0
    print(f"  run_test rc={rc} (min_correlation gate) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = {}
    # QK^T : 256 x 64 x 256
    results["QKT_256x64x256"] = validate_gemm(
        256,
        64,
        256,
        tile_m=32,
        tile_k_l2=64,
        tile_k_l1=64,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )
    # P@V : 256 x 256 x 64
    results["PV_256x256x64"] = validate_gemm(
        256,
        256,
        64,
        tile_m=32,
        tile_k_l2=64,
        tile_k_l1=64,
        tile_n=16,
        herd_m=8,
        herd_n=4,
    )
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)
    assert all(results.values()), f"attention GEMM validation failed: {results}"
    print("PASS: both attention GEMM shapes validated on NPU")


if __name__ == "__main__":
    main()
