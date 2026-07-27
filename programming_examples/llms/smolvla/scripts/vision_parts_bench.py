# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Within-ELF breakdown: every constituent op of vit_ln_qkv / vit_o_ffn, STANDALONE.

Each of the 8 distinct sub-kernels the two fused vision ELFs contain is rebuilt
as its OWN single-launch ELF from the *identical* slice IR (same builder, same
tiles, same `stitch_elf` wrapping — just one slice instead of 7/10) and driven
through the *identical* driver path (`KernelCache.load_and_run` + Profiler), so
the resulting "NPU Run" is directly comparable to the fused ELF's "NPU Run".

That makes `Sigma parts` vs `fused` a clean measurement of the fusion penalty
(hypothesis H1), with no measurement-basis difference to argue about.

  ln     : affine LayerNorm 1024x768                (x2 per layer: LN1, LN2)
  gemm_qkvo : 1024x768x768   drain tm32/tk2 384/tk1 64/tn 96   (x4: q,k,v,o)
  gemm_fc1  : 1024x768x3072  drain tm32/tk2 256/tk1 32/tn 128
  gemm_fc2  : 1024x3072x768  drain tm32/tk2 384/tk1 64/tn 96
  bias768   : broadcast bias-add 1024x768           (x4: q,k,v,o... +fc2 = 4)
  bias3072  : broadcast bias-add 1024x3072          (x1: fc1)
  add768    : residual add 1024x768                 (x2)
  gelu3072  : GELU-tanh 1024x3072                   (x1)

`--sweep-tile-n` additionally re-measures the three GEMM shapes across tile_n
(hypothesis H2: are the fused ELF's tiles registry-optimal?).

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_parts_bench.py
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
CACHE_DIR = str(_HERE / "bench_vision_parts_cache")


def _backend(name, rlts=(2, 2)):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": list(rlts),
    }


# ---------------------------------------------------------------------------
# Build one-slice ELFs from the SAME IR the fused builders splice in
# ---------------------------------------------------------------------------


def _one_slice(func_name, arg_types, ir, extern_syms=frozenset()):
    """Wrap a single sub-kernel IR as a 1-slice stitched ELF module."""
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    base_args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(arg_types)]
    sl = KernelSlice(
        ir, "s", {i: i for i in range(len(arg_types))}, extern_syms=set(extern_syms)
    )
    return stitch_elf(func_name, base_args, [sl])


def _gemm_specs():
    from shared.builders.gemm_builder import (
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from vit_fused_builders import _force_tile_n_suffix

    o = _force_tile_n_suffix(dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")))
    g_, d_ = (
        dict(gemm_registry_config(SEQ, EMB, HID, "bf16", "high")),
        dict(gemm_registry_config(SEQ, HID, EMB, "bf16", "high")),
    )
    o2, g, d = disambiguate_by_tile_n(
        [dict(gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")), g_, d_]
    )
    return {"gemm_qkvo": o, "gemm_fc1": g, "gemm_fc2": d}


def build_parts(cache, which=None, verbose=False):
    """Compile every distinct part ELF (skips those already in the cache)."""
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import _build_gemm_module
    from shared.builders.rms_qkv_bias_rope_multi import _build_bias_add_2d
    from shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from shared.infra.stitching import _wrap_ir_in_launch
    from layer_norm.layer_norm import build_module as build_layer_norm
    from vit_fused_builders import _build_gelu_2d, _gemm_externs

    specs = _gemm_specs()
    todo = []

    # --- GEMMs (identical tiles + identical linked mm.o as the fused ELF) ---
    shapes = {
        "gemm_qkvo": (SEQ, EMB, EMB),
        "gemm_fc1": (SEQ, EMB, HID),
        "gemm_fc2": (SEQ, HID, EMB),
    }
    for name, (m, k, n) in shapes.items():
        if which and name not in which:
            continue
        s = specs[name]
        todo.append(("gemm", name, s, (m, k, n)))

    # --- non-GEMM parts ---
    if not which or "ln" in which:
        todo.append(("ln", "ln", None, None))
    if not which or "bias768" in which:
        todo.append(("bias", "bias768", None, EMB))
    if not which or "bias3072" in which:
        todo.append(("bias", "bias3072", None, HID))
    if not which or "add768" in which:
        todo.append(("add", "add768", None, EMB))
    if not which or "gelu3072" in which:
        todo.append(("gelu", "gelu3072", None, HID))

    for kind, name, spec, extra in todo:
        if name in cache.artifacts:
            print(f"  [cached] {name}")
            continue
        t0 = time.perf_counter()
        if kind == "gemm":
            m, k, n = extra
            compile_gemm_mm(
                tile_m=spec["tile_m"],
                tile_n=spec["tile_n"],
                tile_k_l1=spec["tile_k_l1"],
                sym_suffix=spec["sym_suffix"],
                out_name=spec["obj"],
            )
            ir = str(
                _build_gemm_module(
                    m,
                    k,
                    n,
                    spec["tile_m"],
                    spec["tile_k_l2"],
                    spec["tile_k_l1"],
                    spec["tile_n"],
                    8,
                    4,
                    **spec["build_kwargs"],
                )
            )
            mod = _one_slice(
                name,
                [
                    f"memref<{m}x{k}xbf16>",
                    f"memref<{k}x{n}xbf16>",
                    f"memref<{m}x{n}xbf16>",
                ],
                ir,
                _gemm_externs(spec),
            )
        elif kind == "ln":
            ir = _wrap_ir_in_launch(
                str(build_layer_norm(SEQ, EMB, bfloat16, 16, herd_x=8))
            )
            mod = _one_slice(
                name,
                [
                    f"memref<{SEQ}x{EMB}xbf16>",
                    f"memref<{2*EMB}xbf16>",
                    f"memref<{SEQ}x{EMB}xbf16>",
                ],
                ir,
                {"@zero_vectorized_bf16"},
            )
        elif kind == "bias":
            cols = extra
            ir = str(_build_bias_add_2d(SEQ, cols, cols, bfloat16, 8))
            mod = _one_slice(
                name,
                [
                    f"memref<{SEQ}x{cols}xbf16>",
                    f"memref<{cols}xbf16>",
                    f"memref<{SEQ}x{cols}xbf16>",
                ],
                ir,
            )
        elif kind == "add":
            cols = extra
            ir = str(_build_add_2d_to_2d(SEQ, cols, bfloat16))
            mod = _one_slice(
                name,
                [f"memref<{SEQ}x{cols}xbf16>"] * 3,
                ir,
            )
        elif kind == "gelu":
            cols = extra
            ir = str(_build_gelu_2d(SEQ, cols, 4096, bfloat16, herd_x=8, herd_y=2))
            mod = _one_slice(name, [f"memref<{SEQ}x{cols}xbf16>"] * 2, ir)
        print(f"  compiling {name} ...")
        cache.compile_and_cache(name, mod, {"verbose": verbose, **_backend(name)})
        print(f"    done in {time.perf_counter()-t0:.0f}s")
    cache._save_manifest()


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


def _run_part(cache, name, args, out_idx, static, inter, iters):
    """Drive a part `iters` times; return (median NPU-run ms, median driver ms)."""
    cache.profiler.kernel_breakdowns.pop(name, None)
    cache.profiler.kernel_times.pop(name, None)
    for _ in range(iters):
        res = cache.load_and_run(
            name,
            _backend(name),
            *args,
            output_indices=[out_idx],
            static_input_indices=static,
            intermediate_indices=inter,
            bo_key=name,
        )
    e = cache.profiler.kernel_breakdowns[name][1:]  # drop first (BO alloc/upload)
    ks = sorted(x["kernel_ms"] for x in e)
    ws = sorted(x["write_ms"] for x in e)
    dr = sorted(t * 1e3 for t in cache.profiler.kernel_times[name][1:])
    return (
        float(np.median(ks)),
        float(np.median(ws)),
        float(np.median(dr)),
        res[out_idx],
    )


def measure_parts(cache, iters, rng):
    out = {}

    def z(*shape):
        return np.zeros(int(np.prod(shape)), dtype=bfloat16)

    def r(*shape):
        return (
            rng.standard_normal(int(np.prod(shape))).astype(np.float32).astype(bfloat16)
        )

    specs = _gemm_specs()
    for name, (m, k, n) in {
        "gemm_qkvo": (SEQ, EMB, EMB),
        "gemm_fc1": (SEQ, EMB, HID),
        "gemm_fc2": (SEQ, HID, EMB),
    }.items():
        A, B = r(m, k), r(k, n)
        km, wm, dm, res = _run_part(cache, name, [A, B, z(m, n)], 2, {1}, {2}, iters)
        ref = A.reshape(m, k).astype(np.float32) @ B.reshape(k, n).astype(np.float32)
        got = np.asarray(res, np.float32).reshape(m, n)
        rel = float(np.abs(got - ref).mean() / np.abs(ref).mean())
        gf = 2.0 * m * k * n / (km * 1e-3) / 1e9
        out[name] = dict(
            kernel_ms=km, write_ms=wm, driver_ms=dm, gflops=gf, mean_rel_L1=rel
        )
        print(
            f"  {name:10s} {km*1e3:8.1f} us  {gf:7.0f} GFLOP/s  rel={rel:.3e}  "
            f"(driver {dm*1e3:.0f} us)"
        )

    km, wm, dm, _ = _run_part(
        cache, "ln", [r(SEQ, EMB), r(2 * EMB), z(SEQ, EMB)], 2, {1}, {2}, iters
    )
    out["ln"] = dict(kernel_ms=km, write_ms=wm, driver_ms=dm)
    print(f"  {'ln':10s} {km*1e3:8.1f} us  (driver {dm*1e3:.0f} us)")

    for name, cols in (("bias768", EMB), ("bias3072", HID)):
        km, wm, dm, _ = _run_part(
            cache, name, [r(SEQ, cols), r(cols), z(SEQ, cols)], 2, {1}, {2}, iters
        )
        out[name] = dict(kernel_ms=km, write_ms=wm, driver_ms=dm)
        print(f"  {name:10s} {km*1e3:8.1f} us  (driver {dm*1e3:.0f} us)")

    km, wm, dm, _ = _run_part(
        cache, "add768", [r(SEQ, EMB), r(SEQ, EMB), z(SEQ, EMB)], 2, set(), {2}, iters
    )
    out["add768"] = dict(kernel_ms=km, write_ms=wm, driver_ms=dm)
    print(f"  {'add768':10s} {km*1e3:8.1f} us  (driver {dm*1e3:.0f} us)")

    km, wm, dm, _ = _run_part(
        cache, "gelu3072", [r(SEQ, HID), z(SEQ, HID)], 1, set(), {1}, iters
    )
    out["gelu3072"] = dict(kernel_ms=km, write_ms=wm, driver_ms=dm)
    print(f"  {'gelu3072':10s} {km*1e3:8.1f} us  (driver {dm*1e3:.0f} us)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=21)
    ap.add_argument("--out", default="results/vision_parts.json")
    ap.add_argument("--only", default=None, help="comma list to (re)build")
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="compile the part ELFs and exit (no NPU -> no flock needed)",
    )
    args = ap.parse_args()

    from shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(CACHE_DIR, verbose=False, profiler=Profiler(enabled=True))
    which = set(args.only.split(",")) if args.only else None
    print("[build]")
    build_parts(cache, which=which)
    if args.build_only:
        print("[build-only] done")
        return
    print("\n[measure] median of", args.iters, "invocations (first dropped)")
    rng = np.random.default_rng(0)
    res = measure_parts(cache, args.iters, rng)

    # --- roll up into the two fused groups ---
    ln_qkv = {"ln": 1, "gemm_qkvo": 3, "bias768": 3}
    o_ffn = {
        "gemm_qkvo": 1,
        "bias768": 2,
        "add768": 2,
        "ln": 1,
        "gemm_fc1": 1,
        "bias3072": 1,
        "gelu3072": 1,
        "gemm_fc2": 1,
    }
    print("\n=== Sigma parts (device time, one layer) ===")
    for label, group in (("vit_ln_qkv", ln_qkv), ("vit_o_ffn", o_ffn)):
        tot = 0.0
        print(f"  {label}:")
        for k, mult in group.items():
            v = res[k]["kernel_ms"] * mult
            tot += v
            print(
                f"    {k:10s} x{mult}  {res[k]['kernel_ms']*1e3:7.1f} us -> {v*1e3:8.1f} us"
            )
        print(f"    {'SIGMA':10s}      {tot*1e3:8.1f} us")
        res[f"sigma_{label}"] = tot

    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
