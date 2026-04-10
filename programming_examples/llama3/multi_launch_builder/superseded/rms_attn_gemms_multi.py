#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + Attention GEMMs — 4-launch multi-launch ELF.

Builds a single AIR function with 4 sequential air.launch operations:
  1. RMSNorm      [1,1]   x_in x norm_weight -> normed
  2. Q GEMM       [8,4]   normed x wq -> q_out
  3. K GEMM       [8,4]   normed x wk -> k_out
  4. V GEMM       [8,4]   normed x wv -> v_out

9 func args (4 launches).

Usage:
    python3 rms_attn_gemms_multi.py -p           # print combined MLIR
    python3 rms_attn_gemms_multi.py              # compile + run + validate
    python3 rms_attn_gemms_multi.py --profile    # compile + run + profile
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

from llama3.multi_launch_builder.ffn_full_multi import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _rename_all,
    _fix_launch_func_args,
    _wrap_ir_in_launch,
)


def build_rms_attn_gemms_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    # GEMM tile config (uniform for Q/K/V to avoid hardware state conflicts)
    tile_m=64,
    tile_k_l2=64,
    tile_k_l1=32,
    tile_n=128,
    herd_m=8,
    herd_n=4,
    print_kernels=False,
):
    """Build multi-launch RMSNorm + QKV GEMMs: 4 air.launch ops in one func.

    Args:
        seq_len: Sequence length.
        emb_dim: Embedding dimension.
        kv_dim: Key/Value dimension (n_kv_heads * head_dim).
        print_kernels: If True, print each sub-kernel's MLIR before stitching.

    Returns:
        Module with func @rms_attn_gemms and 9 memref args:
            %arg0: x_in        (seq_len, emb_dim)    - input
            %arg1: norm_weight  (emb_dim,)            - RMSNorm weight
            %arg2: normed       (seq_len, emb_dim)    - RMSNorm output / GEMM input
            %arg3: wq           (emb_dim, emb_dim)    - Q weight
            %arg4: q_out        (seq_len, emb_dim)    - Q output
            %arg5: wk           (emb_dim, kv_dim)     - K weight
            %arg6: k_out        (seq_len, kv_dim)     - K output
            %arg7: wv           (emb_dim, kv_dim)     - V weight
            %arg8: v_out        (seq_len, kv_dim)     - V output
    """
    from llama3.llama3_prefill import _build_gemm_module
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    # Build RMSNorm (needs launch+segment wrapper for multi-launch ELF)
    print("  [1/4] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # Build Q GEMM
    print("  [2/4] Q GEMM...")
    q_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    # Build K GEMM
    print("  [3/4] K GEMM...")
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    # Build V GEMM
    print("  [4/4] V GEMM...")
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
        )
    )

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMM", q_ir),
            ("K GEMM", k_ir),
            ("V GEMM", v_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Stitch: RMSNorm (r) + Q GEMM (q) + K GEMM (k) + V GEMM (v)
    # Arg mapping:
    #   RMSNorm: {0->0, 1->1, 2->2}  (x_in, norm_weight, normed)
    #   Q GEMM:  {0->2, 1->3, 2->4}  (normed, wq, q_out)
    #   K GEMM:  {0->2, 1->5, 2->6}  (normed, wk, k_out)
    #   V GEMM:  {0->2, 1->7, 2->8}  (normed, wv, v_out)
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "r", {0: 0, 1: 1, 2: 2}),
        (q_ir, "q", {0: 2, 1: 3, 2: 4}),
        (k_ir, "k", {0: 2, 1: 5, 2: 6}),
        (v_ir, "v", {0: 2, 1: 7, 2: 8}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Assemble (9 func args, 4 launches)
    combined = "\n".join(maps_all) + f"""
module {{
  func.func @rms_attn_gemms(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{emb_dim}xbf16>,
    %arg4: memref<{seq_len}x{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg6: memref<{seq_len}x{kv_dim}xbf16>,
    %arg7: memref<{emb_dim}x{kv_dim}xbf16>,
    %arg8: memref<{seq_len}x{kv_dim}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


def rms_attn_gemms_reference(x, norm_weight, wq, wk, wv, eps=1e-5):
    """CPU F32 reference: RMSNorm -> Q/K/V GEMMs."""
    x_f32 = x.astype(np.float32)
    w_f32 = norm_weight.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    normed = (x_f32 / rms) * w_f32
    q = normed @ wq.astype(np.float32)
    k = normed @ wk.astype(np.float32)
    v = normed @ wv.astype(np.float32)
    return q.astype(bfloat16), k.astype(bfloat16), v.astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMSNorm + Attn GEMMs multi-launch")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--kv-dim", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    seq_len, emb_dim, kv_dim = args.seq_len, args.emb_dim, args.kv_dim
    print(
        f"RMSNorm + Attn GEMMs: seq_len={seq_len}, emb_dim={emb_dim}, kv_dim={kv_dim}"
    )

    module = build_rms_attn_gemms_module(
        seq_len,
        emb_dim,
        kv_dim,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data
    np.random.seed(42)
    x_in = (np.random.randn(seq_len, emb_dim) * 1.0).astype(bfloat16)
    norm_w = (np.random.randn(emb_dim) * 0.1 + 1.0).astype(bfloat16)
    normed_buf = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    wq = (np.random.randn(emb_dim, emb_dim) * 0.02).astype(bfloat16)
    q_buf = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    wk = (np.random.randn(emb_dim, kv_dim) * 0.02).astype(bfloat16)
    k_buf = np.zeros((seq_len, kv_dim), dtype=bfloat16)
    wv = (np.random.randn(emb_dim, kv_dim) * 0.02).astype(bfloat16)
    v_buf = np.zeros((seq_len, kv_dim), dtype=bfloat16)

    ref_q, ref_k, ref_v = rms_attn_gemms_reference(x_in, norm_w, wq, wk, wv)

    if args.profile:
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="rms_attn_gemms",
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [x_in, norm_w, normed_buf, wq, q_buf, wk, k_buf, wv, v_buf]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        # Warmup
        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()

        # Timed iterations
        times_kernel = []
        for it in range(args.iterations):
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
            times_kernel.append((tk1 - tk0) * 1000)

        # Read outputs and check
        for idx in [4, 6, 8]:
            bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        def read_bo(bo, size):
            return bo.read(size, 0).view(np.int16).view(bfloat16).astype(np.float32)

        npu_q = read_bo(bos[4], sizes[4])
        npu_k = read_bo(bos[6], sizes[6])
        npu_v = read_bo(bos[8], sizes[8])

        corr_q = np.corrcoef(npu_q.flatten(), ref_q.astype(np.float32).flatten())[0, 1]
        corr_k = np.corrcoef(npu_k.flatten(), ref_k.astype(np.float32).flatten())[0, 1]
        corr_v = np.corrcoef(npu_v.flatten(), ref_v.astype(np.float32).flatten())[0, 1]

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel (4 launches): avg={np.mean(times_kernel):.1f}ms  "
            f"min={np.min(times_kernel):.1f}ms  max={np.max(times_kernel):.1f}ms"
        )
        print(f"  Q corr={corr_q:.6f}, K corr={corr_k:.6f}, V corr={corr_v:.6f}")
        ok = corr_q > 0.99 and corr_k > 0.99 and corr_v > 0.99
        print(f"\n  {'PASS' if ok else 'FAIL'}")

    else:
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="rms_attn_gemms",
        )
        exit(
            runner.run_test(
                module,
                inputs=[x_in, norm_w, normed_buf, wq, q_buf, wk, k_buf, wv],
                expected_outputs=[ref_v],
                rtol=0.04,
                atol=4.0,
                min_correlation=0.99,
            )
        )
