# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Within-ELF breakdown by INCREMENTAL PREFIX ELFs (A3-8 study, deliverable §2).

Measuring each sub-kernel of `vit_ln_qkv` / `vit_o_ffn` as a *single-launch* ELF
does not work: a single-launch, multi-M-iteration drain GEMM lowered to ELF is
both ~1.9x slower than the same module lowered to xclbin AND returns garbage on
every invocation after the first (see docs/vision_perf_breakdown.md §"ELF
single-launch pathology"). The shipping multi-launch ELFs are unaffected
(test_full_vit.py PASSes at 0.9906).

So instead we build the SAME fused ELF truncated to its first j launches, for
j = 1..N, and measure each. The marginal cost of launch j *in situ* is
    t(j) - t(j-1)
which is exactly "how much device time does this op cost inside the fused ELF",
measured with no measurement-basis change at all (same driver, same ELF kind,
same herd config).

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_prefix_bench.py
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
CACHE_DIR = str(_HERE / "bench_vision_prefix_cache")

# launch label, output arg index (for output_indices)
LN_QKV_LAUNCHES = ["ln1", "q_gemm", "k_gemm", "v_gemm", "q_bias", "k_bias", "v_bias"]
LN_QKV_OUT = [2, 4, 6, 8, 12, 13, 14]
O_FFN_LAUNCHES = [
    "o_gemm",
    "o_bias",
    "res1",
    "ln2",
    "fc1_gemm",
    "fc1_bias",
    "gelu",
    "fc2_gemm",
    "fc2_bias",
    "res2",
]
O_FFN_OUT = [2, 4, 6, 8, 10, 12, 13, 15, 17, 18]

# Series "b": the same o_ffn slice list with the FIRST launch (o_gemm) dropped,
# so every op lands in the opposite prefix pair. Truncated ELFs only run
# reliably at some prefix lengths (odd j>=3 of o_ffn hang -- an artifact of
# truncating a [2,2] runtime-loop-tiled multi-launch module, not of the shipping
# ELF), so a single series leaves 4 ops merged in pairs. Combining the two
# series' even prefixes determines every op individually.
DROP_FIRST = False


def _backend(name):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


# ---------------------------------------------------------------------------
# Prefix module builders — mirror vit_fused_builders, truncated to j launches
# ---------------------------------------------------------------------------


def build_ln_qkv_prefix(name, j):
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        _wrap_ir_in_launch,
    )
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
    from layer_norm.layer_norm import build_module as build_layer_norm
    from vit_fused_builders import _force_tile_n_suffix, _gemm_externs

    spec = _force_tile_n_suffix(
        dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high"))
    )
    kw = dict(spec["build_kwargs"])
    ln_ir = _wrap_ir_in_launch(str(build_layer_norm(SEQ, EMB, bfloat16, 16, herd_x=8)))
    g_ir = str(
        _build_gemm_module(
            SEQ,
            EMB,
            EMB,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            8,
            4,
            **kw,
        )
    )
    b_ir = str(_build_bias_add_2d(SEQ, EMB, EMB, bfloat16, 8))

    base_args = [
        FuncArg("%arg0", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg1", f"memref<{2*EMB}xbf16>"),
        FuncArg("%arg2", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg3", f"memref<{EMB}x{EMB}xbf16>"),
        FuncArg("%arg4", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg5", f"memref<{EMB}x{EMB}xbf16>"),
        FuncArg("%arg6", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg7", f"memref<{EMB}x{EMB}xbf16>"),
        FuncArg("%arg8", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg9", f"memref<{EMB}xbf16>"),
        FuncArg("%arg10", f"memref<{EMB}xbf16>"),
        FuncArg("%arg11", f"memref<{EMB}xbf16>"),
        FuncArg("%arg12", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg13", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg14", f"memref<{SEQ}x{EMB}xbf16>"),
    ]
    ge = _gemm_externs(spec)
    all_slices = [
        ("ln", ln_ir, {0: 0, 1: 1, 2: 2}, {"@zero_vectorized_bf16"}),
        ("q", g_ir, {0: 2, 1: 3, 2: 4}, ge),
        ("k", g_ir, {0: 2, 1: 5, 2: 6}, ge),
        ("v", g_ir, {0: 2, 1: 7, 2: 8}, ge),
        ("bq", b_ir, {0: 4, 1: 9, 2: 12}, frozenset()),
        ("bk", b_ir, {0: 6, 1: 10, 2: 13}, frozenset()),
        ("bv", b_ir, {0: 8, 1: 11, 2: 14}, frozenset()),
    ]
    return _assemble(name, base_args, all_slices[:j])


def build_o_ffn_prefix(name, j):
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        _wrap_ir_in_launch,
    )
    from shared.builders.gemm_builder import (
        _build_gemm_module,
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
    from shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from layer_norm.layer_norm import build_module as build_layer_norm
    from vit_fused_builders import _build_gelu_2d, _gemm_externs

    o_s, g_s, d_s = disambiguate_by_tile_n(
        [
            dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")),
            dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
            dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
        ]
    )

    def gemm(m, k, n, s):
        return str(
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

    o_ir = gemm(SEQ, EMB, EMB, o_s)
    fc1_ir = gemm(SEQ, EMB, HID, g_s)
    fc2_ir = gemm(SEQ, HID, EMB, d_s)
    b768 = str(_build_bias_add_2d(SEQ, EMB, EMB, bfloat16, 8))
    b3072 = str(_build_bias_add_2d(SEQ, HID, HID, bfloat16, 8))
    add_ir = str(_build_add_2d_to_2d(SEQ, EMB, bfloat16))
    ln_ir = _wrap_ir_in_launch(str(build_layer_norm(SEQ, EMB, bfloat16, 16, herd_x=8)))
    gelu_ir = str(_build_gelu_2d(SEQ, HID, 4096, bfloat16, herd_x=8, herd_y=2))

    base_args = [
        FuncArg("%arg0", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg1", f"memref<{EMB}x{EMB}xbf16>"),
        FuncArg("%arg2", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg3", f"memref<{EMB}xbf16>"),
        FuncArg("%arg4", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg5", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg6", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg7", f"memref<{2*EMB}xbf16>"),
        FuncArg("%arg8", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg9", f"memref<{EMB}x{HID}xbf16>"),
        FuncArg("%arg10", f"memref<{SEQ}x{HID}xbf16>"),
        FuncArg("%arg11", f"memref<{HID}xbf16>"),
        FuncArg("%arg12", f"memref<{SEQ}x{HID}xbf16>"),
        FuncArg("%arg13", f"memref<{SEQ}x{HID}xbf16>"),
        FuncArg("%arg14", f"memref<{HID}x{EMB}xbf16>"),
        FuncArg("%arg15", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg16", f"memref<{EMB}xbf16>"),
        FuncArg("%arg17", f"memref<{SEQ}x{EMB}xbf16>"),
        FuncArg("%arg18", f"memref<{SEQ}x{EMB}xbf16>"),
    ]
    all_slices = [
        ("o", o_ir, {0: 0, 1: 1, 2: 2}, _gemm_externs(o_s)),
        ("bo", b768, {0: 2, 1: 3, 2: 4}, frozenset()),
        ("r1", add_ir, {0: 4, 1: 5, 2: 6}, frozenset()),
        ("ln2", ln_ir, {0: 6, 1: 7, 2: 8}, {"@zero_vectorized_bf16"}),
        ("g", fc1_ir, {0: 8, 1: 9, 2: 10}, _gemm_externs(g_s)),
        ("bg", b3072, {0: 10, 1: 11, 2: 12}, frozenset()),
        ("ge", gelu_ir, {0: 12, 1: 13}, frozenset()),
        ("d", fc2_ir, {0: 13, 1: 14, 2: 15}, _gemm_externs(d_s)),
        ("bd", b768, {0: 15, 1: 16, 2: 17}, frozenset()),
        ("r2", add_ir, {0: 17, 1: 6, 2: 18}, frozenset()),
    ]
    sl = all_slices[1:] if DROP_FIRST else all_slices
    return _assemble(name, base_args, sl[:j])


def _assemble(name, base_args, slices):
    """stitch the truncated slice list; recompute private_from and the
    unreferenced-arg allowance."""
    from shared.infra.stitching import stitch_elf, KernelSlice

    seen = set()
    ks = []
    for prefix, ir, amap, ext in slices:
        key = tuple(sorted(ext))
        first = bool(ext) and key not in seen
        if ext:
            seen.add(key)
        ks.append(
            KernelSlice(
                ir, prefix, dict(amap), extern_syms=set(ext), private_from=first
            )
        )
    referenced = {v for _p, _i, amap, _e in slices for v in amap.values()}
    unref = tuple(i for i in range(len(base_args)) if i not in referenced)
    return stitch_elf(name, base_args, ks, allow_unreferenced_args=unref)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--out", default="results/vision_prefix.json")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--series", default="a", choices=["a", "b"])
    ap.add_argument(
        "--single",
        default=None,
        help="measure ONE prefix tag in this process and append it to --out. "
        "Each ELF needs its own xrt.hw_context and the device runs out of them "
        "after ~10 in one process, so the driver spawns one subprocess per tag.",
    )
    args = ap.parse_args()
    global DROP_FIRST
    DROP_FIRST = args.series == "b"

    from shared.infra.cache import KernelCache, Profiler
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import (
        gemm_registry_config,
        disambiguate_by_tile_n,
    )

    cache = KernelCache(CACHE_DIR, verbose=False, profiler=Profiler(enabled=True))
    cache.load_manifest()

    # Every distinct tile_n-keyed mm.o both groups link (same order as the
    # deployed compile_all_kernels).
    specs = disambiguate_by_tile_n(
        [
            dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")),
            dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
            dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
        ]
    )
    for s in {x["sym_suffix"]: x for x in specs}.values():
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=s["obj"],
        )

    if DROP_FIRST:
        groups = [
            ("vit_o_ffn_b", O_FFN_LAUNCHES[1:], O_FFN_OUT[1:], build_o_ffn_prefix)
        ]
        jrange = (2, 4, 6, 8)  # even prefixes only (odd truncations hang)
    else:
        groups = [
            ("vit_ln_qkv", LN_QKV_LAUNCHES, LN_QKV_OUT, build_ln_qkv_prefix),
            ("vit_o_ffn", O_FFN_LAUNCHES, O_FFN_OUT, build_o_ffn_prefix),
        ]
        jrange = None
    for gname, labels, _outs, builder in groups:
        for j in jrange or range(1, len(labels) + 1):
            tag = f"{gname}_p{j}"
            if tag in cache.artifacts:
                continue
            t0 = time.perf_counter()
            print(f"  building {tag} ({j} launches: {'+'.join(labels[:j])})")
            mod = builder(tag, j)
            cache.compile_and_cache(tag, mod, {"verbose": False, **_backend(tag)})
            print(f"    {time.perf_counter()-t0:.0f}s")
    cache._save_manifest()
    if args.build_only:
        return

    rng = np.random.default_rng(0)

    def rnd(*shape):
        return (
            rng.standard_normal(int(np.prod(shape))).astype(np.float32).astype(bfloat16)
        )

    ln_args = [
        rnd(SEQ, EMB),
        rnd(2 * EMB),
        rnd(SEQ, EMB),
        rnd(EMB, EMB),
        rnd(SEQ, EMB),
        rnd(EMB, EMB),
        rnd(SEQ, EMB),
        rnd(EMB, EMB),
        rnd(SEQ, EMB),
        rnd(EMB),
        rnd(EMB),
        rnd(EMB),
        rnd(SEQ, EMB),
        rnd(SEQ, EMB),
        rnd(SEQ, EMB),
    ]
    offn_args = [
        rnd(SEQ, EMB),
        rnd(EMB, EMB),
        rnd(SEQ, EMB),
        rnd(EMB),
        rnd(SEQ, EMB),
        rnd(SEQ, EMB),
        rnd(SEQ, EMB),
        rnd(2 * EMB),
        rnd(SEQ, EMB),
        rnd(EMB, HID),
        rnd(SEQ, HID),
        rnd(HID),
        rnd(SEQ, HID),
        rnd(SEQ, HID),
        rnd(HID, EMB),
        rnd(SEQ, EMB),
        rnd(EMB),
        rnd(SEQ, EMB),
        rnd(SEQ, EMB),
    ]

    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)

    if args.single:
        gname, j = args.single.rsplit("_p", 1)
        j = int(j)
        if gname == "vit_ln_qkv":
            labels, outs = LN_QKV_LAUNCHES, LN_QKV_OUT
        elif gname == "vit_o_ffn_b":
            labels, outs = O_FFN_LAUNCHES[1:], O_FFN_OUT[1:]
        else:
            labels, outs = O_FFN_LAUNCHES, O_FFN_OUT
        argv = ln_args if gname == "vit_ln_qkv" else offn_args
        for _ in range(args.iters):
            cache.load_and_run(
                args.single,
                _backend(args.single),
                *argv,
                output_indices=[outs[j - 1]],
                bo_key=args.single,
            )
        e = cache.profiler.kernel_breakdowns[args.single][1:]
        km = float(np.median([x["kernel_ms"] for x in e]))
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur[args.single] = {"group": gname, "j": j, "op": labels[j - 1], "total_ms": km}
        p.write_text(json.dumps(cur, indent=2))
        print(f"RESULT {args.single} {km*1e3:.1f} us")
        return

    # In-process loop (fine for a couple of ELFs; use the shell driver for all).
    for gname, labels, outs, _b in groups:
        argv = ln_args if gname == "vit_ln_qkv" else offn_args
        for j in range(1, len(labels) + 1):
            tag = f"{gname}_p{j}"
            for _ in range(args.iters):
                cache.load_and_run(
                    tag,
                    _backend(tag),
                    *argv,
                    output_indices=[outs[j - 1]],
                    bo_key=tag,
                )
            e = cache.profiler.kernel_breakdowns[tag][1:]
            km = float(np.median([x["kernel_ms"] for x in e]))
            cur = json.loads(p.read_text()) if p.exists() else {}
            cur[tag] = {"group": gname, "j": j, "op": labels[j - 1], "total_ms": km}
            p.write_text(json.dumps(cur, indent=2))
            print(f"RESULT {tag} {km*1e3:.1f} us")


if __name__ == "__main__":
    main()
