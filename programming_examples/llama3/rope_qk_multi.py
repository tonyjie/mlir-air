#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE Q + K Multi-Launch — Self-contained test.

Builds a single AIR function with 2 sequential air.herd operations
(RoPE Q → RoPE K), compiles to ELF, runs on NPU, and validates against
CPU F32 reference.

RoPE Q: N = n_heads * seq_len = 32 * 2048 = 65536 rows, head_dim = 64
RoPE K: N = n_kv_heads * seq_len = 8 * 2048 = 16384 rows, head_dim = 64

Usage:
    python3 rope_qk_multi.py -p           # print combined MLIR
    python3 rope_qk_multi.py              # compile + run + validate
"""

import argparse
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.ir import Module, Context
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# MLIR text stitching utilities (adapted from ffn_swiglu/run.py)
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


def _extract_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


def _rename_all(text, prefix):
    """Rename all SSA values and symbols with a unique prefix.

    The external @rope kernel is kept as-is so both herds can call it.
    """
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

    # Symbol names (@herd, @herd_0, etc.) but NOT external kernel @rope
    extern_funcs = {"@rope"}
    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text


def _fix_herd_func_args(text, prefix, arg_map):
    """Fix func-arg references in herd's args() clause after _rename_all.

    After renaming, '%arg0' becomes '%q_arg0'.  The args() clause in the
    air.herd op reads '=%q_arg0' but should reference the outer func's
    '%argN'.  This replaces those back-references.
    """
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rope_qk_module(
    n_heads=32,
    n_kv_heads=8,
    seq_len=2048,
    head_dim=64,
):
    """Build a 2-herd RoPE module: RoPE Q then RoPE K in one function.

    Args:
        n_heads:    Number of query heads (RoPE Q processes n_heads * seq_len rows).
        n_kv_heads: Number of key/value heads (RoPE K processes n_kv_heads * seq_len rows).
        seq_len:    Sequence length.
        head_dim:   Head dimension (must be divisible by 16).

    Returns:
        An mlir.ir.Module containing func @rope_qk with 6 memref args:
            %arg0: Q_in  [q_total], %arg1: Q_lut [q_total]
            %arg2: K_in  [k_total], %arg3: K_lut [k_total]
            %arg4: Q_out [q_total], %arg5: K_out [k_total]
        Outputs are last for XRTRunner compatibility.
    """
    from rope_lut.rope_lut import build_module as build_rope

    N_Q = n_heads * seq_len  # total Q rows (65536 for llama3)
    N_K = n_kv_heads * seq_len  # total K rows (16384 for llama3)

    q_total = N_Q * head_dim
    k_total = N_K * head_dim

    print(f"  [1/2] RoPE Q (N={N_Q}, head_dim={head_dim}, total={q_total})...")
    q_ir = str(build_rope(N_Q, head_dim, bfloat16))

    print(f"  [2/2] RoPE K (N={N_K}, head_dim={head_dim}, total={k_total})...")
    k_ir = str(build_rope(N_K, head_dim, bfloat16))

    # Extract func bodies (between func signature and return).
    # Func arg layout (outputs last, for XRTRunner compatibility):
    #   %arg0: Q_in   %arg1: Q_lut   %arg2: K_in   %arg3: K_lut
    #   %arg4: Q_out  %arg5: K_out
    # Sub-func arg 0 = in, 1 = lut, 2 = out
    # Q herd: {0→0, 1→1, 2→4}
    # K herd: {0→2, 1→3, 2→5}
    bodies = []
    for ir, prefix, arg_map in [
        (q_ir, "q", {0: 0, 1: 1, 2: 4}),
        (k_ir, "k", {0: 2, 1: 3, 2: 5}),
    ]:
        body = _extract_between_func_and_return(ir)
        body = _rename_all(body, prefix)
        body = _fix_herd_func_args(body, prefix, arg_map)
        bodies.append(body)

    # The @rope private declaration is the same for both Q and K (same head_dim).
    # Include it once from the Q module.
    privates = _extract_private_funcs(q_ir)

    combined = f"""module {{
  {"  ".join(p.strip() + chr(10) for p in privates)}  func.func @rope_qk(
    %arg0: memref<{q_total}xbf16>,
    %arg1: memref<{q_total}xbf16>,
    %arg2: memref<{k_total}xbf16>,
    %arg3: memref<{k_total}xbf16>,
    %arg4: memref<{q_total}xbf16>,
    %arg5: memref<{k_total}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
    return
  }}
}}
"""

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

THETA = 10000.0


def _build_lut(n_rows, head_dim, dtype=bfloat16):
    """Build interleaved [cos, sin, cos, sin, ...] RoPE LUT (vectorized)."""
    # freq_i = 1 / (theta ^ (2i / head_dim)), shape: (head_dim//2,)
    i_vals = np.arange(head_dim // 2, dtype=np.float32)
    freqs = 1.0 / (THETA ** (2.0 * i_vals / head_dim))  # (head_dim//2,)
    # angles[r, i] = r * freq_i, shape: (n_rows, head_dim//2)
    rows = np.arange(n_rows, dtype=np.float32)
    angles = np.outer(rows, freqs)  # (n_rows, head_dim//2)
    lut = np.empty((n_rows, head_dim), dtype=np.float32)
    lut[:, 0::2] = np.cos(angles)
    lut[:, 1::2] = np.sin(angles)
    return lut.astype(dtype)


def _apply_rope_cpu(x_2d, lut_2d):
    """Apply RoPE on CPU (F32 precision, vectorized).

    Args:
        x_2d:   (n_rows, head_dim) array
        lut_2d: (n_rows, head_dim) LUT with interleaved [cos, sin, ...]

    Returns:
        output array (n_rows, head_dim) as bfloat16
    """
    x = x_2d.astype(np.float32)
    lut = lut_2d.astype(np.float32)
    # Split into even/odd indices
    x_even = x[:, 0::2]  # x[r, 2i]
    x_odd = x[:, 1::2]  # x[r, 2i+1]
    cos_v = lut[:, 0::2]  # cos values
    sin_v = lut[:, 1::2]  # sin values
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos_v - x_odd * sin_v
    out[:, 1::2] = x_even * sin_v + x_odd * cos_v
    return out.astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default LLAMA-3.2-1B configuration
    N_HEADS = 32
    N_KV_HEADS = 8
    SEQ_LEN = 2048
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(description="RoPE Q+K multi-launch test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print combined MLIR and exit",
    )
    parser.add_argument("--n-heads", type=int, default=N_HEADS)
    parser.add_argument("--n-kv-heads", type=int, default=N_KV_HEADS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
    )
    args = parser.parse_args()

    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    seq_len = args.seq_len
    head_dim = args.head_dim

    N_Q = n_heads * seq_len
    N_K = n_kv_heads * seq_len

    print(
        f"RoPE QK Multi-Launch: n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
        f"seq_len={seq_len}, head_dim={head_dim}"
    )
    print(f"  Q shape: ({N_Q}, {head_dim}) = {N_Q * head_dim} elements")
    print(f"  K shape: ({N_K}, {head_dim}) = {N_K * head_dim} elements")

    module = build_rope_qk_module(n_heads, n_kv_heads, seq_len, head_dim)

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Build test data
    np.random.seed(42)
    q_in = np.random.uniform(-4.0, 4.0, (N_Q, head_dim)).astype(bfloat16)
    k_in = np.random.uniform(-4.0, 4.0, (N_K, head_dim)).astype(bfloat16)

    q_lut = _build_lut(N_Q, head_dim, bfloat16)
    k_lut = _build_lut(N_K, head_dim, bfloat16)

    # CPU reference
    print("Computing CPU reference...")
    q_ref = _apply_rope_cpu(q_in, q_lut)
    k_ref = _apply_rope_cpu(k_in, k_lut)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope_qk",
        )

        # Func signature: (Q_in, Q_lut, K_in, K_lut, Q_out, K_out)
        # Inputs: first 4 args; outputs: last 2 (appended by XRTRunner).
        exit(
            runner.run_test(
                module,
                inputs=[
                    q_in.flatten(),
                    q_lut.flatten(),
                    k_in.flatten(),
                    k_lut.flatten(),
                ],
                expected_outputs=[
                    q_ref.flatten(),
                    k_ref.flatten(),
                ],
                rtol=5e-2,
                atol=5e-2,
                min_correlation=0.99,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
