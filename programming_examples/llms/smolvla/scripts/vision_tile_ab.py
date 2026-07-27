# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""H2: tile_n A/B for the vision GEMMs, INTERLEAVED in one process.

Process-to-process spread on this machine is +-10-15%, which is larger than the
tile_n effect being tested, so measuring config A fully and then config B is
useless. This drives all variants of one GEMM shape round-robin, one invocation
each, for N rounds, and reports the per-variant median. Uses the already-built
`{op}[_tn{n}]_x4` replicate ELFs (4 identical launches, multi-launch ELF = the
regime the deployment runs in); the reported number is the whole x4 ELF, so
divide by 4 for per-launch.

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_tile_ab.py --op gemm_qkvo --tile-ns 96,64,48
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
sys.path.insert(0, str(_HERE / "scripts"))

SEQ, EMB, HID = 1024, 768, 3072
CACHE_DIR = str(_HERE / "bench_vision_repl_cache")
SHAPES = {
    "gemm_qkvo": (SEQ, EMB, EMB),
    "gemm_fc1": (SEQ, EMB, HID),
    "gemm_fc2": (SEQ, HID, EMB),
}
DEPLOYED_TN = {"gemm_qkvo": 96, "gemm_fc1": 128, "gemm_fc2": 96}


def _backend(name):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=list(SHAPES))
    ap.add_argument("--tile-ns", required=True, help="comma list, e.g. 96,64,48")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--out", default="results/vision_tile_ab.json")
    args = ap.parse_args()

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(CACHE_DIR, verbose=False, profiler=Profiler(enabled=True))
    cache.load_manifest()

    m, k, n = SHAPES[args.op]
    tns = [int(x) for x in args.tile_ns.split(",")]
    tags = []
    for tn in tns:
        sfx = "" if tn == DEPLOYED_TN[args.op] else f"_tn{tn}"
        t = f"{args.op}{sfx}_x4"
        if t not in cache.artifacts:
            print(f"  MISSING {t} (build it with vision_replicate_bench --build-only)")
            continue
        tags.append((tn, t))

    rng = np.random.default_rng(0)

    def rnd(shape):
        return (
            rng.standard_normal(int(np.prod(shape))).astype(np.float32).astype(bfloat16)
        )

    A, B = rnd((m, k)), rnd((k, n))
    outs = [rnd((m, n)) for _ in range(4)]
    argv = [A, B] + outs
    ref = A.reshape(m, k).astype(np.float32) @ B.reshape(k, n).astype(np.float32)

    # one warm invocation each (BO alloc + static weight upload)
    for _tn, t in tags:
        res = cache.load_and_run(
            t,
            _backend(t),
            *argv,
            output_indices=[5],
            static_input_indices={0, 1},
            intermediate_indices={2, 3, 4, 5},
            bo_key=t,
        )
        got = np.asarray(res[5], np.float32).reshape(m, n)
        rel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
        print(f"  {t:26s} mean_rel_L1={rel:.3e}")
        cache.profiler.kernel_breakdowns.pop(t, None)

    for _ in range(args.rounds):
        for _tn, t in tags:
            cache.load_and_run(
                t,
                _backend(t),
                *argv,
                output_indices=[5],
                static_input_indices={0, 1},
                intermediate_indices={2, 3, 4, 5},
                bo_key=t,
            )

    print(
        f"\n=== {args.op} {m}x{k}x{n}: tile_n A/B (x4 ELF, {args.rounds} interleaved rounds) ==="
    )
    base = None
    res_json = {}
    for tn, t in tags:
        v = sorted(x["kernel_ms"] for x in cache.profiler.kernel_breakdowns[t])
        med = float(np.median(v))
        per = med / 4.0
        gf = 2.0 * m * k * n / (per * 1e-3) / 1e9
        if tn == DEPLOYED_TN[args.op]:
            base = med
        res_json[tn] = {"x4_ms": med, "per_launch_ms": per, "gflops": gf}
        print(
            f"  tile_n={tn:4d}{' (deployed)' if tn == DEPLOYED_TN[args.op] else '':11s} "
            f"x4={med*1e3:8.1f}us  per-launch={per*1e3:7.1f}us  {gf:6.0f} GFLOP/s"
            f"  [p25 {v[len(v)//4]*1e3:.0f} p75 {v[3*len(v)//4]*1e3:.0f}]"
        )
    if base:
        for tn in res_json:
            res_json[tn]["speedup_vs_deployed"] = base / res_json[tn]["x4_ms"]
            if tn != DEPLOYED_TN[args.op]:
                print(f"  tile_n={tn} vs deployed: {base/res_json[tn]['x4_ms']:.3f}x")
    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[args.op] = res_json
    p.write_text(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
