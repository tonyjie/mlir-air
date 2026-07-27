# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-op IN-SITU device cost inside a multi-launch ELF (A3-8 study, §2).

For each sub-kernel of the two fused vision ELFs, build two ELFs containing
**2 and 4 identical copies** of that launch (each writing its own output arg)
and measure both. Then

    cost_per_launch = (t4 - t2) / 2          fixed ELF overhead cancels
    elf_overhead    = t2 - 2 * cost_per_launch

This avoids both traps found earlier:
  * a SINGLE-launch multi-M-iteration GEMM lowered to ELF is ~1.9x slower than
    the same module lowered to xclbin and returns garbage after the first
    invocation, so 1-launch ELFs cannot be used as the "standalone" reference;
  * truncated prefixes of the real fused ELF hang at some prefix lengths.

Both ELFs here are multi-launch and even-length, i.e. in the same regime the
shipping ELFs run in.

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_replicate_bench.py --op ...
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

SEQ, EMB, HID = 1024, 768, 3072
CACHE_DIR = str(_HERE / "bench_vision_repl_cache")
COPIES = (2, 4)

# op -> (n_shared_inputs, out_rows, out_cols)
OPS = [
    "gemm_qkvo",
    "gemm_fc1",
    "gemm_fc2",
    "ln",
    "bias768",
    "bias3072",
    "add768",
    "gelu",
]


def _backend(name):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


TILE_N_OVERRIDE = None


def _op_pieces(op):
    """-> (slice_ir, extern_syms, shared_arg_types, out_type, n_shared)"""
    from shared.builders.gemm_builder import (
        _build_gemm_module,
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
    from shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from shared.infra.stitching import _wrap_ir_in_launch
    from layer_norm.layer_norm import build_module as build_layer_norm
    from vit_fused_builders import _build_gelu_2d, _gemm_externs

    o_s, g_s, d_s = disambiguate_by_tile_n(
        [
            dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")),
            dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
            dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
        ]
    )
    if op.startswith("gemm"):
        m, k, n, s = {
            "gemm_qkvo": (SEQ, EMB, EMB, o_s),
            "gemm_fc1": (SEQ, EMB, HID, g_s),
            "gemm_fc2": (SEQ, HID, EMB, d_s),
        }[op]
        ir = str(
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
            )
        )
        return (
            ir,
            _gemm_externs(s),
            [f"memref<{m}x{k}xbf16>", f"memref<{k}x{n}xbf16>"],
            f"memref<{m}x{n}xbf16>",
            2,
            s,
        )
    if op == "ln":
        ir = _wrap_ir_in_launch(str(build_layer_norm(SEQ, EMB, bfloat16, 16, herd_x=8)))
        return (
            ir,
            {"@zero_vectorized_bf16"},
            [f"memref<{SEQ}x{EMB}xbf16>", f"memref<{2*EMB}xbf16>"],
            f"memref<{SEQ}x{EMB}xbf16>",
            2,
            None,
        )
    if op in ("bias768", "bias3072"):
        c = EMB if op == "bias768" else HID
        ir = str(_build_bias_add_2d(SEQ, c, c, bfloat16, 8))
        return (
            ir,
            frozenset(),
            [f"memref<{SEQ}x{c}xbf16>", f"memref<{c}xbf16>"],
            f"memref<{SEQ}x{c}xbf16>",
            2,
            None,
        )
    if op == "add768":
        ir = str(_build_add_2d_to_2d(SEQ, EMB, bfloat16))
        return (
            ir,
            frozenset(),
            [f"memref<{SEQ}x{EMB}xbf16>", f"memref<{SEQ}x{EMB}xbf16>"],
            f"memref<{SEQ}x{EMB}xbf16>",
            2,
            None,
        )
    if op == "gelu":
        ir = str(_build_gelu_2d(SEQ, HID, 4096, bfloat16, herd_x=8, herd_y=2))
        return (
            ir,
            frozenset(),
            [f"memref<{SEQ}x{HID}xbf16>"],
            f"memref<{SEQ}x{HID}xbf16>",
            1,
            None,
        )
    raise ValueError(op)


def build_repl(name, op, copies):
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    ir, ext, shared_ty, out_ty, n_shared, _s = _op_pieces(op)
    base_args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(shared_ty)]
    base_args += [FuncArg(f"%arg{n_shared+i}", out_ty) for i in range(copies)]
    slices = []
    for i in range(copies):
        amap = {a: a for a in range(n_shared)}
        amap[n_shared] = n_shared + i
        slices.append(
            KernelSlice(ir, f"c{i}", amap, extern_syms=set(ext), private_from=(i == 0))
        )
    return stitch_elf(name, base_args, slices)


def _args_for(op, copies, rng):
    _ir, _e, shared_ty, _o, n_shared, _s = _op_pieces(op)

    def rnd(shape):
        return (
            rng.standard_normal(int(np.prod(shape))).astype(np.float32).astype(bfloat16)
        )

    shapes = {
        "gemm_qkvo": [(SEQ, EMB), (EMB, EMB)],
        "gemm_fc1": [(SEQ, EMB), (EMB, HID)],
        "gemm_fc2": [(SEQ, HID), (HID, EMB)],
        "ln": [(SEQ, EMB), (2 * EMB,)],
        "bias768": [(SEQ, EMB), (EMB,)],
        "bias3072": [(SEQ, HID), (HID,)],
        "add768": [(SEQ, EMB), (SEQ, EMB)],
        "gelu": [(SEQ, HID)],
    }[op]
    outshape = {
        "gemm_qkvo": (SEQ, EMB),
        "gemm_fc1": (SEQ, HID),
        "gemm_fc2": (SEQ, EMB),
        "ln": (SEQ, EMB),
        "bias768": (SEQ, EMB),
        "bias3072": (SEQ, HID),
        "add768": (SEQ, EMB),
        "gelu": (SEQ, HID),
    }[op]
    return [rnd(s) for s in shapes] + [rnd(outshape) for _ in range(copies)], n_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=OPS)
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--out", default="results/vision_replicate.json")
    ap.add_argument("--tile-n", type=int, default=None)
    args = ap.parse_args()
    global TILE_N_OVERRIDE
    TILE_N_OVERRIDE = args.tile_n
    suffix = f"_tn{args.tile_n}" if args.tile_n else ""

    from shared.infra.cache import KernelCache, Profiler
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import (
        gemm_registry_config,
        disambiguate_by_tile_n,
    )

    cache = KernelCache(CACHE_DIR, verbose=False, profiler=Profiler(enabled=True))
    cache.load_manifest()
    specs = disambiguate_by_tile_n(
        [
            dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")),
            dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
            dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
        ]
    )
    if TILE_N_OVERRIDE is not None:
        from vit_fused_builders import _force_tile_n_suffix

        specs = [
            _force_tile_n_suffix({**dict(x), "tile_n": TILE_N_OVERRIDE}) for x in specs
        ]
    for s in {x["sym_suffix"]: x for x in specs}.values():
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=s["obj"],
        )

    for c in COPIES:
        tag = f"{args.op}{suffix}_x{c}"
        if tag in cache.artifacts:
            continue
        t0 = time.perf_counter()
        print(f"  building {tag}")
        cache.compile_and_cache(
            tag, build_repl(tag, args.op, c), {"verbose": False, **_backend(tag)}
        )
        print(f"    {time.perf_counter()-t0:.0f}s")
        cache._save_manifest()
    if args.build_only:
        return

    rng = np.random.default_rng(0)
    res = {}
    for c in COPIES:
        tag = f"{args.op}{suffix}_x{c}"
        argv, n_shared = _args_for(args.op, c, rng)
        for _ in range(args.iters):
            cache.load_and_run(
                tag,
                _backend(tag),
                *argv,
                output_indices=[n_shared + c - 1],
                static_input_indices=set(range(n_shared)),
                intermediate_indices=set(range(n_shared, n_shared + c)),
                bo_key=tag,
            )
        e = cache.profiler.kernel_breakdowns[tag][1:]
        res[c] = float(np.median([x["kernel_ms"] for x in e]))
        print(f"  {tag}: {res[c]*1e3:8.1f} us")

    per = (res[4] - res[2]) / 2.0
    overhead = res[2] - 2 * per
    print(
        f"RESULT {args.op}{suffix} per_launch={per*1e3:8.1f} us  elf_overhead={overhead*1e3:7.1f} us"
    )
    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[args.op + suffix] = {
        "t2_ms": res[2],
        "t4_ms": res[4],
        "per_launch_ms": per,
        "elf_overhead_ms": overhead,
    }
    p.write_text(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
