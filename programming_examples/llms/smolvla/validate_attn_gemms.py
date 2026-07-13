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
    # FP32 reference: full-precision matmul of the bf16-rounded inputs. This is
    # the SAME reference every registered GEMM row uses for mean_rel_L1 (the
    # device path is f32-accumulate + single epilogue bf16 cast, so the FP32
    # matmul is the ground truth; the bf16 output rounding is the error we
    # measure). NOT a bf16-cast reference -- that would hide the cast error.
    C_ref_f32 = A.astype(np.float32) @ B.astype(np.float32)

    # PROJECT-STANDARD PRECISION METRIC (matches all other GEMM_bf16_in_bf16_out
    # rows): mean_rel_L1 = mean|out - ref| / mean|ref| vs the FP32 reference,
    # gated at the bf16-out GEMM tier (~9e-3, same as every other bf16-out row).
    # Pearson correlation is ALSO reported (scale/shift-invariant sanity check)
    # but mean_rel_L1 is the gate -- correlation alone cannot catch a systematic
    # scale/bias, so it does not meet the registry bar on its own.
    MEAN_REL_L1_TIER = 1.2e-2  # bf16-out GEMM tier ceiling (other rows: 9.3-9.9e-3)

    runner = XRTRunner(
        verbose=False,
        output_format="xclbin",
        runtime_loop_tiling_sizes=[4, 4],
    )
    # Capture the device output so we can compute mean_rel_L1 ourselves. We drive
    # the backend directly (run_test only returns pass/fail), reusing the harness
    # compile+run path.
    actual = _run_and_get_output(runner, mod, [A, B], np.zeros((m, n), bfloat16))
    actual_f32 = actual.reshape(m, n).astype(np.float32)

    mean_rel_L1 = float(
        np.mean(np.abs(actual_f32 - C_ref_f32)) / np.mean(np.abs(C_ref_f32))
    )
    corr = float(np.corrcoef(actual_f32.flatten(), C_ref_f32.flatten())[0, 1])
    ok = np.isfinite(mean_rel_L1) and mean_rel_L1 <= MEAN_REL_L1_TIER
    print(
        f"  mean_rel_L1 = {mean_rel_L1:.4e} (tier <= {MEAN_REL_L1_TIER:.1e}) "
        f"| corr = {corr:.6f} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, mean_rel_L1, corr


def _run_and_get_output(runner, mlir_module, inputs, out_placeholder):
    """Compile + run on NPU and return the raw device output array.

    Mirrors XRTRunner.run_test's compile+invoke path (inputs + output
    placeholder, filelock around load+run) but returns the actual output
    (run_test only returns a pass/fail code). out_placeholder fixes the output
    shape/dtype and is passed as the output slot.
    """
    import os
    import tempfile
    import filelock
    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=False,
        omit_while_true_loop=runner.omit_while_true_loop,
        runtime_loop_tiling_sizes=runner.runtime_loop_tiling_sizes,
        output_format="xclbin",
    )
    compiled = backend.compile(mlir_module)
    expanded = list(inputs) + [out_placeholder]
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        module_function = backend.load(compiled)
        actual_outputs = module_function(*expanded)
    backend.unload()
    actual_outputs = list(actual_outputs[len(inputs) :])
    return np.asarray(actual_outputs[0], dtype=out_placeholder.dtype)


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
    for k, (ok, mrl1, corr) in results.items():
        print(
            f"  {k}: {'PASS' if ok else 'FAIL'} | "
            f"mean_rel_L1={mrl1:.4e} corr={corr:.6f}"
        )
    print("=" * 60)
    assert all(
        ok for ok, _, _ in results.values()
    ), f"attention GEMM validation failed (mean_rel_L1 tier): {results}"
    print("PASS: both attention GEMM shapes meet the bf16 mean_rel_L1 tier on NPU")


if __name__ == "__main__":
    main()
