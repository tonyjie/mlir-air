#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN Full Multi-Launch — RMSNorm + FFN + Residual Add in one ELF.

Builds a single AIR function with 6 sequential air.launch operations:
  1. RMSNorm         [1,1]   res1 x ffn_norm_weight -> normed2
  2. Gate GEMM       [8,4]   normed2 x w_gate -> gate_buf
  3. Up GEMM         [8,4]   normed2 x w_up -> up_buf
  4. SwiGLU          [8,1]   SiLU(gate_buf) x up_buf -> swiglu_buf
  5. Down GEMM       [8,4]   swiglu_buf x w_down -> down_out
  6. Eltwise Add     [8,1]   res1_flat + down_flat -> output

11 func args (6 launches). The eltwise add inputs (res1, down_out) are
created via memref.collapse_shape from the 2D args, avoiding extra BOs.

Usage:
    python3 ffn_full_multi.py -p           # print combined MLIR
    python3 ffn_full_multi.py              # compile + run + validate
    python3 ffn_full_multi.py --profile    # compile + run + profile
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
# MLIR text stitching utilities (shared with ffn_swiglu/run.py)
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


def _extract_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


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
    extern_funcs = {"@silu_and_mul_bf16", "@zero_vectorized_bf16", "@matmul_bf16"}
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


def _wrap_ir_in_launch(mlir_text):
    """Wrap a module whose func body contains bare herds in air.launch.

    Transforms:
        func.func @name(%arg0: T0, %arg1: T1, %arg2: T2) {
            <body with bare air.herd>
            return
        }
    Into:
        func.func @name(%arg0: T0, %arg1: T1, %arg2: T2) {
            air.launch () in () args(%arg3=%arg0, %arg4=%arg1, %arg5=%arg2) : T0, T1, T2 {
                <body with herd refs remapped to %arg3, %arg4, %arg5>
            }
            return
        }

    This is needed because multi-launch ELF modules require each kernel to be
    wrapped in air.launch, but RMSNorm and Eltwise Add generate bare herds.
    """
    lines = mlir_text.split("\n")

    # Find the public func signature
    func_line_idx = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            func_line_idx = i
            break
    if func_line_idx is None:
        return mlir_text

    func_line = lines[func_line_idx]

    # Check if body already has air.launch (skip wrapping)
    body_text = "\n".join(lines[func_line_idx + 1 :])
    if "air.launch" in body_text:
        return mlir_text

    # Parse func args: extract (%arg0: type0, %arg1: type1, ...)
    # The func signature may span multiple lines if complex, but for our
    # simple kernels it's on one line.
    sig_match = re.search(r"func\.func @\w+\(([^)]*)\)", func_line)
    if not sig_match:
        return mlir_text

    args_str = sig_match.group(1)
    # Parse individual args
    func_args = []
    for arg in args_str.split(","):
        arg = arg.strip()
        if not arg:
            continue
        parts = arg.split(":")
        name = parts[0].strip()
        typ = ":".join(parts[1:]).strip()
        func_args.append((name, typ))

    n_args = len(func_args)

    # Find the body (between func line and return)
    body_start = func_line_idx + 1
    body_end = None
    for i in range(len(lines) - 1, body_start, -1):
        if lines[i].strip() == "return":
            body_end = i
            break

    body_lines = lines[body_start:body_end]
    body_text = "\n".join(body_lines)

    # Find the max %argN index used in the body to avoid conflicts.
    # The herd uses %arg3-%arg9 internally, so launch args must start higher.
    existing_args = [int(m) for m in re.findall(r"%arg(\d+)", body_text)]
    max_existing = max(existing_args) if existing_args else n_args - 1
    launch_arg_start = max_existing + 1

    # Build the air.launch args clause with safe arg indices
    launch_args = ", ".join(
        f"%arg{launch_arg_start + i}={func_args[i][0]}" for i in range(n_args)
    )
    launch_types = ", ".join(func_args[i][1] for i in range(n_args))

    # In the body, remap func arg references to launch arg references.
    # Process in reverse order of arg index to avoid partial replacements.
    for i in range(n_args - 1, -1, -1):
        old_name = func_args[i][0]  # e.g. %arg0
        new_name = f"%arg{launch_arg_start + i}"  # e.g. %arg10
        body_text = re.sub(re.escape(old_name) + r"(?!\w)", new_name, body_text)

    # Reconstruct the module
    new_lines = lines[:body_start]
    new_lines.append(f"    air.launch () in () args({launch_args}) : {launch_types} {{")
    # Indent body by 2 more spaces
    for line in body_text.split("\n"):
        new_lines.append("  " + line)
    new_lines.append("    }")
    new_lines.extend(lines[body_end:])

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_ffn_full_module(
    seq_len=2048,
    emb_dim=2048,
    hidden_dim=8192,
    # Gate/Up GEMM tile config
    gate_tile_m=64,
    gate_tile_k_l2=64,
    gate_tile_k_l1=32,
    gate_tile_n=128,
    gate_herd_m=8,
    gate_herd_n=4,
    # Down GEMM tile config
    down_tile_m=64,
    down_tile_k_l2=256,
    down_tile_k_l1=32,
    down_tile_n=64,
    down_herd_m=8,
    down_herd_n=4,
    # SwiGLU config
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    # Add config
    add_tile_n=2048,
    add_herd_x=8,
    add_herd_y=1,
    print_kernels=False,
):
    """Build multi-launch FFN Full: 6 air.launch ops in one func.

    Combines: RMSNorm + Gate GEMM + Up GEMM + SwiGLU + Down GEMM + Eltwise Add

    Args:
        print_kernels: If True, print each sub-kernel's MLIR before stitching.
    """
    from llama3.llama3_prefill import _build_gemm_module
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from eltwise_add.eltwise_add import build_module as build_add

    # Import silu_and_mul from ffn_swiglu directory
    import importlib.util

    _silu_path = os.path.join(
        os.path.dirname(__file__), "ffn_swiglu", "silu_and_mul.py"
    )
    _spec = importlib.util.spec_from_file_location("silu_and_mul", _silu_path)
    _silu_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_silu_mod)
    build_swiglu = _silu_mod.build_module_2d

    n_total = seq_len * emb_dim

    # Build each kernel independently
    # RMSNorm and Eltwise Add generate bare herds (no air.launch wrapper).
    # Multi-launch ELF requires each kernel in an air.launch, so we wrap them.
    print("  [1/6] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, emb_dim, bfloat16, 16)))

    print("  [2/6] Gate GEMM...")
    gate_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [3/6] Up GEMM...")
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [4/6] SwiGLU...")
    swiglu_ir = str(
        build_swiglu(
            seq_len,
            hidden_dim,
            swiglu_tile_n,
            bfloat16,
            herd_x=swiglu_herd_x,
            herd_y=swiglu_herd_y,
        )
    )

    print("  [5/6] Down GEMM...")
    down_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            down_tile_m,
            down_tile_k_l2,
            down_tile_k_l1,
            down_tile_n,
            down_herd_m,
            down_herd_n,
        )
    )

    print("  [6/6] Eltwise Add...")
    add_ir = _wrap_ir_in_launch(
        str(
            build_add(
                n_total,
                add_tile_n,
                bfloat16,
                vector_size=16,
                herd_x=add_herd_x,
                herd_y=add_herd_y,
            )
        )
    )

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Gate GEMM", gate_ir),
            ("Up GEMM", up_ir),
            ("SwiGLU", swiglu_ir),
            ("Down GEMM", down_ir),
            ("Eltwise Add", add_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Extract, rename, remap
    # Arg mapping (each original module has args 0,1,2):
    #   RMSNorm:    {0->0, 1->1, 2->2}  (res1, weight, normed2)
    #   Gate GEMM:  {0->2, 1->3, 2->4}  (normed2, w_gate, gate_buf)
    #   Up GEMM:    {0->2, 1->5, 2->6}  (normed2, w_up, up_buf)
    #   SwiGLU:     {0->4, 1->6, 2->7}  (gate_buf, up_buf, swiglu_buf)
    #   Down GEMM:  {0->7, 1->8, 2->9}  (swiglu_buf, w_down, down_out)
    #   Eltwise Add: uses collapse_shape of arg0/arg9 + arg10 (output)
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (rms_ir, "r", {0: 0, 1: 1, 2: 2}),
        (gate_ir, "g", {0: 2, 1: 3, 2: 4}),
        (up_ir, "u", {0: 2, 1: 5, 2: 6}),
        (swiglu_ir, "s", {0: 4, 1: 6, 2: 7}),
        (down_ir, "d", {0: 7, 1: 8, 2: 9}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    # Eltwise Add: needs special handling because its inputs are collapse_shape
    # views of 2D buffers (arg0=res1 and arg9=down_out), not separate func args.
    add_body = _extract_between_func_and_return(add_ir)
    add_maps = _extract_affine_maps(add_ir)
    add_body = _rename_all(add_body, "a")
    add_maps = [_rename_all(m, "a") for m in add_maps]
    # Map: add arg0 -> %res1_flat, add arg1 -> %down_flat, add arg2 -> %arg10
    add_body = add_body.replace("=%a_arg0,", "=%res1_flat,")
    add_body = add_body.replace("=%a_arg0)", "=%res1_flat)")
    add_body = add_body.replace("=%a_arg1,", "=%down_flat,")
    add_body = add_body.replace("=%a_arg1)", "=%down_flat)")
    add_body = add_body.replace("=%a_arg2,", "=%arg10,")
    add_body = add_body.replace("=%a_arg2)", "=%arg10)")
    bodies.append(add_body)
    maps_all.extend(add_maps)

    # Collect private func declarations (SwiGLU has the external kernel)
    privates = _extract_private_funcs(swiglu_ir)

    # collapse_shape ops to create 1D aliases for the eltwise add
    collapse_ops = f"""\
    %res1_flat = memref.collapse_shape %arg0 [[0, 1]] : memref<{seq_len}x{emb_dim}xbf16> into memref<{n_total}xbf16>
    %down_flat = memref.collapse_shape %arg9 [[0, 1]] : memref<{seq_len}x{emb_dim}xbf16> into memref<{n_total}xbf16>"""

    # Assemble the combined module (11 func args instead of 13)
    combined = "\n".join(maps_all) + f"""
module {{
  {"  ".join(p.strip() + chr(10) for p in privates)}  func.func @ffn_full(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{seq_len}x{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg4: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg5: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg6: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg7: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg8: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg9: memref<{seq_len}x{emb_dim}xbf16>,
    %arg10: memref<{n_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
{bodies[4]}
{collapse_ops}
{bodies[5]}
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


def ffn_full_reference(x, ffn_norm_weight, w_gate, w_up, w_down, eps=1e-5):
    """CPU F32 reference: RMSNorm -> Gate -> Up -> SwiGLU -> Down -> Residual Add.

    Args:
        x: (seq_len, emb_dim) input (residual state)
        ffn_norm_weight: (emb_dim,) RMSNorm weight
        w_gate: (emb_dim, hidden_dim) gate projection weight
        w_up: (emb_dim, hidden_dim) up projection weight
        w_down: (hidden_dim, emb_dim) down projection weight
        eps: RMSNorm epsilon

    Returns:
        output: (seq_len, emb_dim) = x + down_proj(SwiGLU(gate, up))
    """
    x_f32 = x.astype(np.float32)
    w_f32 = ffn_norm_weight.astype(np.float32)

    # RMSNorm
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    normed = (x_f32 / rms) * w_f32

    # Gate + Up projections
    gate = normed @ w_gate.astype(np.float32)
    up = normed @ w_up.astype(np.float32)

    # SwiGLU: SiLU(gate) * up
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    swiglu = (gate * sigmoid) * up

    # Down projection
    down = swiglu @ w_down.astype(np.float32)

    # Residual add
    output = x_f32 + down
    return output.astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFN Full multi-launch test")
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
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument(
        "--iterations", type=int, default=5, help="Profiling iterations"
    )
    args = parser.parse_args()

    seq_len, emb_dim, hidden_dim = args.seq_len, args.emb_dim, args.hidden_dim
    n_total = seq_len * emb_dim
    print(
        f"FFN Full Multi-Launch: seq_len={seq_len}, emb_dim={emb_dim}, "
        f"hidden_dim={hidden_dim}"
    )

    module = build_ffn_full_module(
        seq_len, emb_dim, hidden_dim, print_kernels=args.print_kernels
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data
    np.random.seed(42)
    res1 = (np.random.randn(seq_len, emb_dim) * 1.0).astype(bfloat16)
    ffn_norm_weight = (np.random.randn(emb_dim) * 0.1 + 1.0).astype(bfloat16)
    normed2_buf = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    w_gate = (np.random.randn(emb_dim, hidden_dim) * 0.1).astype(bfloat16)
    gate_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    w_up = (np.random.randn(emb_dim, hidden_dim) * 0.1).astype(bfloat16)
    up_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    swiglu_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    w_down = (np.random.randn(hidden_dim, emb_dim) * 0.01).astype(bfloat16)
    down_out = np.zeros((seq_len, emb_dim), dtype=bfloat16)

    # Output buffer for residual add (1D)
    output_buf = np.zeros(n_total, dtype=bfloat16)

    # CPU reference
    output_ref = ffn_full_reference(res1, ffn_norm_weight, w_gate, w_up, w_down)

    if args.profile:
        # Profile mode: compile, load, run N iterations
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="ffn_full",
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [
            res1,  # arg0: res1 (2D)
            ffn_norm_weight,  # arg1: ffn_norm_weight (1D)
            normed2_buf,  # arg2: normed2 (2D)
            w_gate,  # arg3: w_gate (2D)
            gate_buf,  # arg4: gate_buf (2D)
            w_up,  # arg5: w_up (2D)
            up_buf,  # arg6: up_buf (2D)
            swiglu_buf,  # arg7: swiglu_buf (2D)
            w_down,  # arg8: w_down (2D)
            down_out,  # arg9: down_out (2D)
            output_buf,  # arg10: output (1D)
        ]
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
        times_kernel, times_total = [], []
        for it in range(args.iterations):
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
            # Only read back output (arg10)
            bos[10].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            t1 = time.perf_counter()
            times_kernel.append((tk1 - tk0) * 1000)
            times_total.append((t1 - t0) * 1000)

        # Check correctness
        output_data = bos[10].read(sizes[10], 0).view(np.int16).view(bfloat16)
        output_data = output_data.reshape(seq_len, emb_dim).astype(np.float32)
        ref_flat = output_ref.astype(np.float32).flatten()
        corr = np.corrcoef(output_data.flatten(), ref_flat)[0, 1]

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel (6 launches): avg={np.mean(times_kernel):.1f}ms  "
            f"min={np.min(times_kernel):.1f}ms  max={np.max(times_kernel):.1f}ms"
        )
        print(
            f"  Total (write+run+read): avg={np.mean(times_total):.1f}ms  "
            f"min={np.min(times_total):.1f}ms  max={np.max(times_total):.1f}ms"
        )
        print(f"  Host overhead: {np.mean(times_total) - np.mean(times_kernel):.1f}ms")
        print(f"  Correlation: {corr:.6f}")
        status = "PASS" if corr > 0.99 else "FAIL"
        print(f"\n  {status} (corr={corr:.6f})")

    else:
        # Correctness test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="ffn_full",
        )
        exit(
            runner.run_test(
                module,
                inputs=[
                    res1,
                    ffn_norm_weight,
                    normed2_buf,
                    w_gate,
                    gate_buf,
                    w_up,
                    up_buf,
                    swiglu_buf,
                    w_down,
                    down_out,
                ],
                expected_outputs=[output_ref.reshape(-1)],
                rtol=0.04,
                atol=4.0,
                min_correlation=0.99,
            )
        )
