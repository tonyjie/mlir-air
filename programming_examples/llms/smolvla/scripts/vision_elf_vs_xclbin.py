# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The same GEMM module, lowered to ELF vs to xclbin, driven by the SAME driver.

The SmolVLA deployment runs every kernel as a multi-launch **ELF** (the xclbin
path mis-runs multi-launch modules). The kernel registry's `drain` throughput
numbers, by contrast, are measured on the **xclbin** path
(`matrix_multiplication/bf16_in_bf16_out/run.py:1156`:
`use_elf = ... and method == "fused-cast"`). This script measures both formats
of the identical module through `KernelCache.load_and_run`, interleaved in one
process, so the comparison has no basis difference at all.

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_elf_vs_xclbin.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

CACHE_DIR = str(_HERE / "bench_fmt_cache")
SEQ, EMB, HID = 1024, 768, 3072
SHAPES = {
    "qkvo": (SEQ, EMB, EMB),
    "fc1": (SEQ, EMB, HID),
    "fc2": (SEQ, HID, EMB),
}


def bk(fmt, name="matmul_bf16"):
    return {
        "omit_while_true_loop": False,
        "output_format": fmt,
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--out", default="results/vision_elf_vs_xclbin.json")
    ap.add_argument(
        "--fmt",
        default=None,
        choices=["elf", "xclbin"],
        help="measure ONE format in this process. Interleaving an\n                    ELF hw_context with an xclbin hw_context costs more device\n                    time than the effect under test (measured), so the two\n                    formats must be measured in separate processes.",
    )
    args = ap.parse_args()

    from shared.infra.cache import KernelCache, Profiler
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import (
        _build_gemm_module,
        gemm_registry_config,
        disambiguate_by_tile_n,
    )

    cache = KernelCache(CACHE_DIR, verbose=False, profiler=Profiler(enabled=True))
    cache.load_manifest()
    specs = dict(
        zip(
            ("qkvo", "fc1", "fc2"),
            disambiguate_by_tile_n(
                [
                    dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")),
                    dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
                    dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
                ]
            ),
        )
    )
    for s in {x["sym_suffix"]: x for x in specs.values()}.values():
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=s["obj"],
        )

    out = {}
    for label, (m, k, n) in SHAPES.items():
        s = specs[label]
        tags = []
        for fmt in ("elf", "xclbin"):
            tag = f"{label}_{fmt}"
            if tag not in cache.artifacts:
                print(f"  building {tag}")
                cache.compile_and_cache(
                    tag,
                    _build_gemm_module(
                        m,
                        k,
                        n,
                        s["tile_m"],
                        s["tile_k_l2"],
                        s["tile_k_l1"],
                        s["tile_n"],
                        8,
                        4,
                        **dict(s["build_kwargs"]),
                    ),
                    {"verbose": False, **bk(fmt)},
                )
                cache._save_manifest()
            if args.fmt is None or fmt == args.fmt:
                tags.append((fmt, tag))

        rng = np.random.default_rng(0)
        A = rng.standard_normal(m * k).astype(np.float32).astype(bfloat16)
        B = rng.standard_normal(k * n).astype(np.float32).astype(bfloat16)
        ref = A.reshape(m, k).astype(np.float32) @ B.reshape(k, n).astype(np.float32)

        for _fmt, tag in tags:
            cache.profiler.kernel_breakdowns.pop(tag, None)
        for r in range(args.rounds):
            for _fmt, tag in tags:
                C = np.zeros(m * n, dtype=bfloat16)
                res = cache.load_and_run(
                    tag,
                    bk(_fmt),
                    A,
                    B,
                    C,
                    output_indices=[2],
                    static_input_indices={1},
                    bo_key=tag,
                )
                if r == args.rounds - 1:
                    got = np.asarray(res[2], np.float32).reshape(m, n)
                    rel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
                    out.setdefault(label, {}).setdefault(_fmt, {})["mean_rel_L1"] = rel

        print(
            f"\n=== {label} {m}x{k}x{n} (tile_n={s['tile_n']}), {args.rounds} interleaved rounds ==="
        )
        for _fmt, tag in tags:
            v = sorted(
                x["kernel_ms"] for x in cache.profiler.kernel_breakdowns[tag][1:]
            )
            med = float(np.median(v))
            gf = 2.0 * m * k * n / (med * 1e-3) / 1e9
            out[label][_fmt].update(npu_run_ms=med, gflops=gf)
            print(
                f"  {_fmt:7s} NPU-run {med*1e3:8.1f} us  {gf:6.0f} GFLOP/s  "
                f"mean_rel_L1={out[label][_fmt]['mean_rel_L1']:.3e}"
            )
        if "elf" in out[label] and "xclbin" in out[label]:
            r = out[label]["elf"]["npu_run_ms"] / out[label]["xclbin"]["npu_run_ms"]
            out[label]["elf_penalty"] = r
            print(f"  -> ELF is {r:.2f}x slower than xclbin for the SAME module")

    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    for kk, vv in out.items():
        cur.setdefault(kk, {}).update(vv)
    p.write_text(json.dumps(cur, indent=2))
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
