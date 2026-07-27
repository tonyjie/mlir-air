# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Isolate: is the 1st invocation of a multi-M-iteration drain GEMM ELF correct
and every LATER one wrong?  (M=1024 drain tile_m=32 herd_m=8 -> 4 M-iterations.)

Each call re-uploads ALL buffers (no static/intermediate flags) so buffer state
cannot explain a difference; only the run-object / instruction-stream state can.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from shared.infra.cache import KernelCache, Profiler  # noqa: E402

CACHE = str(_HERE / "bench_gemm_drive_cache")
K = N = 768


def bk():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "matmul_bf16",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def main():
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import _build_gemm_module

    cache = KernelCache(CACHE, verbose=False, profiler=Profiler(enabled=True))
    compile_gemm_mm(
        tile_m=32,
        tile_n=96,
        tile_k_l1=64,
        sym_suffix="_m32_n96",
        out_name="mm_m32_n96.o",
    )
    rng = np.random.default_rng(0)
    for M in (256, 1024):
        tag = f"g{M}"
        if tag not in cache.artifacts:
            mod = _build_gemm_module(
                M,
                K,
                N,
                32,
                384,
                64,
                96,
                8,
                4,
                external_bf16_out=True,
                sym_suffix="_m32_n96",
                link_with_name="mm_m32_n96.o",
            )
            cache.compile_and_cache(tag, mod, {"verbose": False, **bk()})
        A = rng.standard_normal(M * K).astype(np.float32).astype(bfloat16)
        B = rng.standard_normal(K * N).astype(np.float32).astype(bfloat16)
        ref = A.reshape(M, K).astype(np.float32) @ B.reshape(K, N).astype(np.float32)
        print(f"--- M={M} ({M//(32*8)} M-launch iterations) ---")
        for call in range(5):
            C = np.zeros(M * N, dtype=bfloat16)
            res = cache.load_and_run(
                tag,
                bk(),
                A,
                B,
                C,
                output_indices=[2],
                bo_key=f"{tag}_shared",  # same BOs, everything re-uploaded each call
            )
            got = np.asarray(res[2], np.float32).reshape(M, N)
            rel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
            blk = (
                np.abs(got - ref).reshape(M // 32, 32, N).mean(axis=(1, 2))
                / np.abs(ref).mean()
            )
            bad = np.nonzero(blk > 0.1)[0]
            print(
                f"  call {call}: rel={rel:.3e}  bad 32-row blocks {len(bad)}/{M//32}"
                f"  {bad[:6].tolist()}{'...' if len(bad) > 6 else ''}"
            )


if __name__ == "__main__":
    main()
