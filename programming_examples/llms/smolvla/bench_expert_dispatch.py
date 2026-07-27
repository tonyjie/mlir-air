"""MEASUREMENT: per-dispatch host overhead on the real deployment path.

The action-expert feasibility projection is dominated by a single number:
what does ONE NPU dispatch cost end-to-end from the Python driver, on top of
the kernel's own device time?

Two costs are distinguished here, and they are NOT the same number:

  1. DEVICE time -- `xrt.run.start()` + `wait2()` only, args already bound,
     no BO sync. This is what every kernel_registry GFLOPS number and the
     GEMM harness's `--perf-iters` report (air/backend/xrt.py:573-584).
  2. DRIVER time -- one full `KernelCache.load_and_run(...)` call as the
     SmolVLA backbone/vision drivers actually issue it
     (llms/shared/infra/cache.py:427+): filelock acquire, write the
     non-static inputs into their BOs + sync-to-device, construct a fresh
     `xrt.run`, `set_arg` every buffer, start+wait, sync the outputs back,
     zero-copy map them out, profiler bookkeeping, filelock release.

DRIVER - DEVICE = the per-dispatch overhead the fusion work exists to remove.

This script measures both for the action expert's own q-projection GEMM
(M=64 (50 action tokens padded), K=720, N=960, bf16-in/bf16-out, drain,
tile_m=16 / tile_k_l2=144 / tile_k_l1=48 / tile_n=80, herd 4x4) -- the exact
config the sweep picked in results/expert_gemm.csv -- plus a no-op-sized
control so the fixed part can be separated from the payload-size part.

Weights are marked `static_input_indices` and the output `intermediate_indices`,
exactly as the deployed backbone marks them, so the measured overhead is the
BEST case the current driver can do (only the activation is re-uploaded).

Usage (needs the NPU lock; compile is cached in bench_expert_dispatch_cache/):
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 bench_expert_dispatch.py
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_PROG = _LLMS.parent
for p in (str(_LLMS), str(_PROG)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ml_dtypes import bfloat16

from shared.infra.cache import KernelCache, Profiler
from shared.infra.external_kernels import compile_gemm_mm
from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm

# (label, M, K, N, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n)
SHAPES = [
    ("q_proj_64x720x960", 64, 720, 960, 16, 144, 48, 80, 4, 4),
    ("tiny_64x32x64", 64, 32, 32, 16, 32, 32, 16, 4, 2),
]

GEMM_BACKEND = {"omit_while_true_loop": False}


def build(cache, label, M, K, N, tm, tk2, tk1, tn, hm, hn):
    # mm.o must be rebuilt per (tile_m, tile_n, tile_k_l1) triple -- the dims are
    # baked in as -DDIM_*. compile_and_cache stages it into air_project/.
    compile_gemm_mm(tile_m=tm, tile_n=tn, tile_k_l1=tk1, out_name="mm.o")
    mod = build_gemm(
        M,
        K,
        N,
        tm,
        tk2,
        tk1,
        tn,
        hm,
        hn,
        bfloat16,
        bfloat16,
        arch="aie2p",
        emit_external_call=True,  # == harness `--method drain`
        drain_chunks=1,
    )
    cache.compile_and_cache(label, mod, GEMM_BACKEND)


def bench_driver(cache, label, M, K, N, iters, warmup):
    """Time full KernelCache.load_and_run calls (the deployed driver path)."""
    a = (np.random.randn(M, K) / np.sqrt(K)).astype(bfloat16)
    b = (np.random.randn(K, N) / np.sqrt(K)).astype(bfloat16)
    c = np.zeros((M, N), dtype=bfloat16)

    def call():
        return cache.load_and_run(
            label,
            GEMM_BACKEND,
            a,
            b,
            c,
            output_indices=[2],
            static_input_indices=[1],  # weight uploaded once
            intermediate_indices=[2],  # output BO never re-uploaded
            bo_key=label,
        )

    for _ in range(warmup):
        call()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return ts


def summarize_breakdown(prof, label):
    rows = prof.kernel_breakdowns.get(label, [])
    if not rows:
        return None
    keys = ("write_ms", "kernel_ms", "read_ms")
    return {k: statistics.median([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cache-dir", default=str(_HERE / "bench_expert_dispatch_cache"))
    args = ap.parse_args()

    prof = Profiler(enabled=True)
    cache = KernelCache(Path(args.cache_dir), verbose=True, profiler=prof)

    results = {}
    for spec in SHAPES:
        label = spec[0]
        print(f"\n=== {label} ===")
        build(cache, *spec)
        M, K, N = spec[1], spec[2], spec[3]
        ts = bench_driver(cache, label, M, K, N, args.iters, args.warmup)
        bd = summarize_breakdown(prof, label)
        results[label] = (ts, bd, M, K, N)
        print(
            f"  driver wall/call: median {statistics.median(ts):.3f} ms | "
            f"min {min(ts):.3f} | p90 {sorted(ts)[int(0.9*len(ts))]:.3f} | "
            f"mean {statistics.mean(ts):.3f}"
        )
        if bd:
            print(
                f"  breakdown (median ms): write={bd['write_ms']:.3f} "
                f"kernel(start+wait)={bd['kernel_ms']:.3f} read={bd['read_ms']:.3f}"
            )

    print(f"\n{'='*88}")
    print("PER-DISPATCH COST ON THE DEPLOYED DRIVER PATH (NPU2, this run)")
    print(f"{'='*88}")
    print(
        f"| {'kernel':22s} | {'driver ms':>9s} | {'kernel ms':>9s} | "
        f"{'write ms':>8s} | {'read ms':>8s} | {'overhead ms':>11s} |"
    )
    for label, (ts, bd, M, K, N) in results.items():
        med = statistics.median(ts)
        if bd:
            oh = med - bd["kernel_ms"]
            print(
                f"| {label:22s} | {med:9.3f} | {bd['kernel_ms']:9.3f} | "
                f"{bd['write_ms']:8.3f} | {bd['read_ms']:8.3f} | {oh:11.3f} |"
            )
    print(
        "\noverhead = driver wall - (start+wait). It is the part fusion removes:\n"
        "fusing K ops into 1 ELF pays it once instead of K times."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
