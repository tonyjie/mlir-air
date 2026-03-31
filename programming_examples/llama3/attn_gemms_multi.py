#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attention GEMMs Multi-Launch — Self-contained test.

Builds a single AIR function with 3 sequential air.launch operations
(Q GEMM + K GEMM + V GEMM), compiles to ELF, runs on NPU,
and validates against CPU F32 reference.

Architecture:
    func @attn_gemms(
        %arg0: memref<seq_len x emb_dim x bf16>,    # input (normed)
        %arg1: memref<emb_dim x emb_dim x bf16>,    # wq
        %arg2: memref<seq_len x emb_dim x bf16>,    # q_out
        %arg3: memref<emb_dim x kv_dim x bf16>,     # wk
        %arg4: memref<seq_len x kv_dim x bf16>,     # k_out
        %arg5: memref<emb_dim x kv_dim x bf16>,     # wv
        %arg6: memref<seq_len x kv_dim x bf16>,     # v_out
    ):
        air.launch 1: Q GEMM   input x wq -> q_out
        air.launch 2: K GEMM   input x wk -> k_out
        air.launch 3: V GEMM   input x wv -> v_out
        return

Usage:
    python3 attn_gemms_multi.py -p           # print combined MLIR
    python3 attn_gemms_multi.py              # compile + run + validate
    python3 attn_gemms_multi.py --profile    # compile + run + profile
"""

import argparse
import os
import re
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.ir import *
from air.dialects.air import *
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# MLIR text stitching utilities (same as ffn_swiglu/run.py)
# ---------------------------------------------------------------------------


def _extract_between_func_and_return(mlir_text):
    """Extract func body (between func signature and return)."""
    lines = mlir_text.split("\n")
    body_start = body_end = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = i + 1
    for i in range(len(lines) - 1, body_start, -1):
        if lines[i].strip() == "return":
            body_end = i
            break
    return "\n".join(lines[body_start:body_end])


def _extract_affine_maps(mlir_text):
    return [l for l in mlir_text.split("\n") if l.startswith("#map")]


def _rename_all(text, prefix):
    """Rename all SSA values, affine maps, and symbols with a unique prefix."""
    # Affine maps (longest first)
    for name in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)

    # SSA word values (%argN, %cN, %allocN, etc.)
    for name in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)

    # SSA numbered values (%0, %1, ...)
    for name in sorted(
        set(re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = text.replace(name, f"%{prefix}_n{name[1:]}")

    # Symbol names (@seg, @herd) but NOT external kernel functions
    extern_funcs = {"@matmul_bf16"}
    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text


def _fix_launch_func_args(text, prefix, arg_map):
    """Fix func-arg references in launch's args() clause after _rename_all."""
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_attn_gemms_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    # Q GEMM tile config — uses K/V config by default for uniform tiling.
    # Mixed configs (e.g. Q: tile_k_l2=256, tile_n=64) cause NPU timeout
    # on repeated ELF runs due to a hardware state conflict between launches
    # with different L2 tiling layouts. Single-shot execution works fine.
    q_tile_m=64,
    q_tile_k_l2=64,
    q_tile_k_l1=32,
    q_tile_n=128,
    q_herd_m=8,
    q_herd_n=4,
    # K/V GEMM tile config
    kv_tile_m=64,
    kv_tile_k_l2=64,
    kv_tile_k_l1=32,
    kv_tile_n=128,
    kv_herd_m=8,
    kv_herd_n=4,
    print_kernels=False,
):
    """Build multi-launch attention GEMMs: 3 air.launch ops in one func.

    Args:
        print_kernels: If True, print each sub-kernel's MLIR before stitching.
    """
    from llama3.llama3_prefill import _build_gemm_module

    # Build each kernel independently
    print("  [1/3] Q GEMM...")
    q_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            q_tile_m,
            q_tile_k_l2,
            q_tile_k_l1,
            q_tile_n,
            q_herd_m,
            q_herd_n,
        )
    )

    print("  [2/3] K GEMM...")
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            kv_tile_m,
            kv_tile_k_l2,
            kv_tile_k_l1,
            kv_tile_n,
            kv_herd_m,
            kv_herd_n,
        )
    )

    print("  [3/3] V GEMM...")
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            kv_tile_m,
            kv_tile_k_l2,
            kv_tile_k_l1,
            kv_tile_n,
            kv_herd_m,
            kv_herd_n,
        )
    )

    if print_kernels:
        for name, ir in [
            ("Q GEMM", q_ir),
            ("K GEMM", k_ir),
            ("V GEMM", v_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Extract, rename, remap
    # Q GEMM: original {0->0, 1->1, 2->2} (input, wq, q_out)
    # K GEMM: original {0->0, 1->3, 2->4} (input, wk, k_out)
    # V GEMM: original {0->0, 1->5, 2->6} (input, wv, v_out)
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (q_ir, "q", {0: 0, 1: 1, 2: 2}),
        (k_ir, "k", {0: 0, 1: 3, 2: 4}),
        (v_ir, "v", {0: 0, 1: 5, 2: 6}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Assemble
    combined = "\n".join(maps_all) + f"""
module {{
  func.func @attn_gemms(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg4: memref<{seq_len}x{kv_dim}xbf16>,
    %arg5: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg6: memref<{seq_len}x{kv_dim}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def attn_gemms_reference(x, wq, wk, wv):
    """CPU F32 reference: q = x @ wq, k = x @ wk, v = x @ wv."""
    x_f32 = x.astype(np.float32)
    q = (x_f32 @ wq.astype(np.float32)).astype(bfloat16)
    k = (x_f32 @ wk.astype(np.float32)).astype(bfloat16)
    v = (x_f32 @ wv.astype(np.float32)).astype(bfloat16)
    return q, k, v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention GEMMs multi-launch test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--print-kernels",
        action="store_true",
        help="Print each sub-kernel's MLIR before stitching",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile kernel execution"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--kv-dim", type=int, default=512)
    parser.add_argument(
        "--iterations", type=int, default=5, help="Profiling iterations"
    )
    args = parser.parse_args()

    seq_len, emb_dim, kv_dim = args.seq_len, args.emb_dim, args.kv_dim
    print(
        f"Attn GEMMs Multi-Launch: seq_len={seq_len}, emb_dim={emb_dim}, kv_dim={kv_dim}"
    )

    module = build_attn_gemms_module(
        seq_len, emb_dim, kv_dim, print_kernels=args.print_kernels
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data
    np.random.seed(42)
    x = (np.random.randn(seq_len, emb_dim) * 1.0).astype(bfloat16)
    wq = (np.random.randn(emb_dim, emb_dim) * 0.1).astype(bfloat16)
    wk = (np.random.randn(emb_dim, kv_dim) * 0.1).astype(bfloat16)
    wv = (np.random.randn(emb_dim, kv_dim) * 0.1).astype(bfloat16)
    q_buf = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    k_buf = np.zeros((seq_len, kv_dim), dtype=bfloat16)
    v_buf = np.zeros((seq_len, kv_dim), dtype=bfloat16)

    q_ref, k_ref, v_ref = attn_gemms_reference(x, wq, wk, wv)

    if args.profile:
        # Profile mode: compile once, reload + run each iteration.
        # ELF format has a known issue where large GEMMs (e.g. 2048x2048x2048)
        # produce stale results on repeated runs without reload. Reloading the
        # kernel each iteration adds ~150ms overhead but ensures correctness.
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="attn_gemms",
        )
        artifact = backend.compile(module)

        inputs = [x, wq, q_buf, wk, k_buf, wv, v_buf]
        sizes = [a.size * a.itemsize for a in inputs]

        # Timed iterations (reload kernel each time for correctness)
        times_kernel, times_total = [], []
        for it in range(args.iterations):
            # Reload kernel context each iteration
            with filelock.FileLock("/tmp/npu.lock"):
                invoker = backend.load(artifact)

            bos = [xrt.ext.bo(backend.device, s) for s in sizes]

            t0 = time.perf_counter()
            for i, a in enumerate(inputs):
                bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            tk0 = time.perf_counter()
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()
            tk1 = time.perf_counter()
            # Read all 3 outputs for profiling
            for out_idx in [2, 4, 6]:
                bos[out_idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            t1 = time.perf_counter()
            times_kernel.append((tk1 - tk0) * 1000)
            times_total.append((t1 - t0) * 1000)

            backend.unload()

        # Correctness from last iteration
        q_data = bos[2].read(sizes[2], 0).view(np.int16).view(bfloat16)
        q_data = q_data.reshape(seq_len, emb_dim).astype(np.float32)
        k_data = bos[4].read(sizes[4], 0).view(np.int16).view(bfloat16)
        k_data = k_data.reshape(seq_len, kv_dim).astype(np.float32)
        v_data = bos[6].read(sizes[6], 0).view(np.int16).view(bfloat16)
        v_data = v_data.reshape(seq_len, kv_dim).astype(np.float32)

        corr_q = np.corrcoef(q_data.flatten(), q_ref.astype(np.float32).flatten())[0, 1]
        corr_k = np.corrcoef(k_data.flatten(), k_ref.astype(np.float32).flatten())[0, 1]
        corr_v = np.corrcoef(v_data.flatten(), v_ref.astype(np.float32).flatten())[0, 1]

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel (3 launches): avg={np.mean(times_kernel):.1f}ms  "
            f"min={np.min(times_kernel):.1f}ms  max={np.max(times_kernel):.1f}ms"
        )
        print(
            f"  Total (write+run+read): avg={np.mean(times_total):.1f}ms  "
            f"min={np.min(times_total):.1f}ms  max={np.max(times_total):.1f}ms"
        )
        print(f"  Host overhead: {np.mean(times_total) - np.mean(times_kernel):.1f}ms")
        print(f"\n  Correlation:")
        print(f"    Q GEMM: {corr_q:.6f}")
        print(f"    K GEMM: {corr_k:.6f}")
        print(f"    V GEMM: {corr_v:.6f}")
        min_corr = min(corr_q, corr_k, corr_v)
        status = "PASS" if min_corr > 0.999 else "FAIL"
        print(f"\n  {status} (min_corr={min_corr:.6f})")

    else:
        # Correctness test — XRTRunner only reads back the last buffer (v_out)
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="attn_gemms",
        )
        exit(
            runner.run_test(
                module,
                inputs=[x, wq, q_buf, wk, k_buf, wv],
                expected_outputs=[v_ref.reshape(-1)],
                rtol=0.04,
                atol=4.0,
                min_correlation=0.999,
            )
        )
