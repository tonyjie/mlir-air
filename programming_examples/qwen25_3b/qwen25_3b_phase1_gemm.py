#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-3B Phase 1 — standalone NPU2 cosine verification of the 4
GEMM shapes the deployment uses.

Tile configs chosen subject to:
  * `N % (tile_n × herd_n) == 0`  (silent-corruption trap — qwen25 LESSON 2)
  * `K % tile_k_l2 == 0`          (DMA chunking)
  * `K % tile_k_l1 == 0`          (inner accumulation)
  * L1 budget per tile ≤ ~32 KB BF16 (well under 64 KB)
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "_llm_shared"))

from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module
from air.backend.xrt_runner import XRTRunner

# Qwen2.5-3B production GEMM shape table.
# emb_dim=2048 (1024-aligned), hidden=11008 (NOT 1024-aligned but tile-friendly:
# 11008 = 2^7 × 86, divisors include 64×4=256 → 11008/256 = 43 iterations).
SHAPES = {
    "qo": dict(
        name="Q/O",
        m=2048,
        k=2048,
        n=2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    ),
    "kv": dict(
        name="K/V",
        m=2048,
        k=2048,
        n=256,
        tile_m=64,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    ),
    "gateup": dict(
        name="Gate/Up",
        m=2048,
        k=2048,
        n=11008,
        tile_m=64,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    ),
    "down": dict(
        name="Down",
        m=2048,
        k=11008,
        n=2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    ),
}

RTOL = 8e-2
ATOL = 2.0
MIN_CORR = 0.999


def gemm_reference(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)


def run_one(shape_key, verbose=False):
    cfg = SHAPES[shape_key]
    m, k, n = cfg["m"], cfg["k"], cfg["n"]
    tiles = cfg["herd_m"] * cfg["herd_n"]
    print(f"\n{'='*70}")
    print(
        f"  {cfg['name']:7s}  M={m}  K={k}  N={n}  herd={cfg['herd_m']}x{cfg['herd_n']}={tiles} tiles"
    )
    print(
        f"  tile_m={cfg['tile_m']}  tile_k_l2={cfg['tile_k_l2']}  tile_k_l1={cfg['tile_k_l1']}  tile_n={cfg['tile_n']}"
    )
    # Sanity-check the silent-corruption trap.
    assert (
        n % (cfg["tile_n"] * cfg["herd_n"]) == 0
    ), f"{cfg['name']}: N={n} not divisible by tile_n*herd_n={cfg['tile_n']*cfg['herd_n']}"
    assert k % cfg["tile_k_l2"] == 0
    assert m % (cfg["tile_m"] * cfg["herd_m"]) == 0
    print(f"{'='*70}")

    module = _build_gemm_module(
        m,
        k,
        n,
        cfg["tile_m"],
        cfg["tile_k_l2"],
        cfg["tile_k_l1"],
        cfg["tile_n"],
        cfg["herd_m"],
        cfg["herd_n"],
    )

    np.random.seed(42)
    a = (np.random.randn(m, k) * 1.0).astype(bfloat16)
    b = (np.random.randn(k, n) * 0.1).astype(bfloat16)
    c_ref = gemm_reference(a, b)

    runner = XRTRunner(
        verbose=verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="matmul_bf16",
        runtime_loop_tiling_sizes=[2, 2],
    )
    rc = runner.run_test(
        module,
        inputs=[a, b],
        expected_outputs=[c_ref],
        rtol=RTOL,
        atol=ATOL,
        min_correlation=MIN_CORR,
    )
    status = "PASS" if rc == 0 else "FAIL"
    print(f"  XRTRunner verdict: rc={rc} → {status}")
    return rc == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B Phase 1 GEMM cosine verification"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="all",
        choices=["all", "qo", "kv", "gateup", "down"],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    keys = list(SHAPES.keys()) if args.shape == "all" else [args.shape]
    results = {}
    for key in keys:
        results[key] = run_one(key, verbose=args.verbose)

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    for key, ok in results.items():
        cfg = SHAPES[key]
        marker = "PASS" if ok else "FAIL"
        print(f"  {cfg['name']:7s}  ({cfg['m']}×{cfg['k']}×{cfg['n']}): {marker}")

    if not all(results.values()):
        sys.exit(1)
