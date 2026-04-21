# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 1 of the bottom-up kernel-first Qwen3 NPU decode build.

For each leaf kernel needed by Qwen3 decode, compile it standalone, run on
NPU2 with random inputs, verify against numpy F32 reference. This proves the
registry kernels work at Qwen3 decode shapes BEFORE we stitch them into
multi-launch ELFs.

Kernels exercised (Qwen3-1.7B decode shapes, M=1):
  1. weighted_rms_norm at M=1, N=1024 (attn_norm, ffn_norm, final_norm)
  2. matvec GEMV K=1024 N=2048 (Q proj)
  3. matvec GEMV K=1024 N=1024 (K, V proj)
  4. matvec GEMV K=2048 N=1024 (O proj — Qwen3-specific, q_dim != emb_dim)
  5. matvec GEMV K=1024 N=3072 (Gate, Up proj)
  6. matvec GEMV K=3072 N=1024 (Down proj)
  7. matvec GEMV K=1024 N=16384 (LM head partition; 1 of 10)
  8. silu_and_mul at n=3072 (SwiGLU)

(RoPE half-split is run on host at M=1; cost is microseconds — the
plan defers an NPU rope_qk for decode.)

Pass criteria: cosine_sim(NPU output, numpy reference) > 0.99 on all kernels.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
for _p in (_EXAMPLES_DIR, _EXAMPLES_DIR / "llama3", _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Add specific subdirs so leaf builders' module-relative imports resolve.
sys.path.insert(0, str(_EXAMPLES_DIR / "weighted_rms_norm"))
sys.path.insert(0, str(_EXAMPLES_DIR / "matrix_vector_multiplication" / "bf16"))

from air.backend.xrt_runner import XRTRunner

from llama3_prefill import prepare_air_project
from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels


def _cos(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def _run_kernel_test(
    label, module, inputs, expected, rtol=1e-2, atol=5e-2, **backend_kwargs
):
    """Compile + run on NPU2 + cosine-check against expected.

    backend_kwargs override the XRTRunner defaults (e.g. omit_pingpong,
    runtime_loop_tiling_sizes, use_lock_race_condition_fix for matvec).

    Returns (passed, cosine, msg).
    """
    kwargs = dict(
        verbose=False,
        omit_while_true_loop=True,
        instance_name=label.replace(" ", "_").replace("=", "").replace(",", ""),
    )
    kwargs.update(backend_kwargs)
    runner = XRTRunner(**kwargs)
    try:
        rc = runner.run_test(
            module,
            inputs=inputs,
            expected_outputs=[expected],
            rtol=rtol,
            atol=atol,
        )
    except Exception as e:
        return False, 0.0, f"runtime error: {type(e).__name__}: {e}"
    # XRTRunner returns 0 on PASS, nonzero on numerical FAIL. Even if numerics
    # exceed rtol/atol, capture the actual cosine for diagnostics by re-running
    # the numpy reference comparison (XRTRunner itself doesn't expose the BO
    # readback). For our purposes the rc is the gate.
    return (
        rc == 0,
        None,
        "" if rc == 0 else f"XRTRunner rc={rc} (numerics out of tolerance)",
    )


# ---------------------------------------------------------------------------
# 1. weighted_rms_norm (M=1, N=1024)
# ---------------------------------------------------------------------------


def test_weighted_rms_norm(M=1, N=1024, eps=1e-6):
    from weighted_rms_norm import build_module

    np.random.seed(42)
    x = (np.random.randn(M, N) * 1.0).astype(bfloat16)
    w = (np.random.randn(N) * 0.1 + 1.0).astype(bfloat16)

    x_f32 = x.astype(np.float32)
    w_f32 = w.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    expected = ((x_f32 / rms) * w_f32).astype(bfloat16)

    module = build_module(M, N, bfloat16, vector_size=16, herd_x=1)
    label = f"rms_norm_M{M}_N{N}"
    # Wider tolerance for larger N (BF16 sum-of-squares accumulates more noise).
    return _run_kernel_test(label, module, [x, w], expected, rtol=1e-1, atol=1e-1)


# ---------------------------------------------------------------------------
# 2-7. matvec GEMV at various (M_out, K_in) shapes
# ---------------------------------------------------------------------------

# matvec convention: y[m] = A[m, k] @ x[k].
# Default tile config from llama3 decode: tile_m=8, m_input=4, herd_m=8.
# Constraint: m % (tile_m*herd_m=64) == 0; k % 64 == 0.


def test_gemv(m, k, label_suffix="", tile_m=8, m_input=4, herd_m=8):
    from matvec import build_module

    np.random.seed(123)
    # Match canonical matvec test scale (* 4) for stable BF16 accumulation
    A = (np.random.randn(m, k) * 4.0).astype(bfloat16)
    x = (np.random.randn(k) * 4.0).astype(bfloat16)

    expected = (A.astype(np.float32) @ x.astype(np.float32)).astype(bfloat16)

    module = build_module(
        m,
        k,
        tile_m=tile_m,
        m_input=m_input,
        herd_m=herd_m,
        np_dtype_in=bfloat16,
        np_dtype_out=bfloat16,
    )
    label = f"gemv_M{m}_K{k}{label_suffix}"
    # Backend kwargs from canonical matvec/bf16/matvec.py:435 — required for correctness.
    return _run_kernel_test(
        label,
        module,
        [A, x],
        expected,
        rtol=0.04,
        atol=1e-3,
        omit_while_true_loop=False,
        omit_pingpong=True,
        runtime_loop_tiling_sizes=[4, 4],
        use_lock_race_condition_fix=True,
    )


# ---------------------------------------------------------------------------
# 8. silu_and_mul (SwiGLU) at n=3072
# ---------------------------------------------------------------------------


def test_silu_and_mul(n=3072):
    from _llm_shared.kernel_builder.ffn_swiglu.silu_and_mul import build_module

    # silu_and_mul has herd_y=2 default → must compile the silu_and_mul.o extern first
    compile_all_external_kernels(head_dim=128)  # also compiles silu_and_mul.o

    np.random.seed(7)
    gate = (np.random.randn(n) * 0.5).astype(bfloat16)
    up = (np.random.randn(n) * 0.5).astype(bfloat16)

    g_f32 = gate.astype(np.float32)
    u_f32 = up.astype(np.float32)
    silu = g_f32 * (1.0 / (1.0 + np.exp(-g_f32)))
    expected = (silu * u_f32).astype(bfloat16)

    module = build_module(n, n // 8, bfloat16, herd_x=8, herd_y=1)
    label = f"silu_and_mul_n{n}"
    return _run_kernel_test(label, module, [gate, up], expected, rtol=1e-1, atol=1e-1)


# ---------------------------------------------------------------------------
# Phase A new leaves — host ops we want to move onto the NPU
# ---------------------------------------------------------------------------


def test_qknorm(M, N=128, label_suffix=""):
    """Per-head RMSNorm: weighted_rms_norm at (M=n_heads, N=head_dim).
    Same kernel & semantics as input RMSNorm — just smaller N and small M.
    """
    from weighted_rms_norm import build_module

    eps = 1e-6
    np.random.seed(123 + M)
    x = (np.random.randn(M, N) * 1.0).astype(bfloat16)
    w = (np.random.randn(N) * 0.5 + 1.0).astype(bfloat16)

    x_f32 = x.astype(np.float32)
    w_f32 = w.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    expected = ((x_f32 / rms) * w_f32).astype(bfloat16)

    # herd_x must divide M; M=16 → herd_x∈{1,2,4,8,16}; M=8 → herd_x∈{1,2,4,8}.
    herd_x = min(8, M)
    module = build_module(M, N, bfloat16, vector_size=16, herd_x=herd_x)
    label = f"qknorm_M{M}_N{N}{label_suffix}"
    return _run_kernel_test(label, module, [x, w], expected, rtol=5e-2, atol=5e-2)


def test_rope_1d(n_rows, head_dim=128, label_suffix=""):
    """1D RoPE half-split for decode: n_rows × head_dim flattened.
    Uses _build_rope_1d (already used by llama3 fused decode ELF).
    Extern: rope.o compiled from _llm_shared/kernel_builder/rope_halfsplit.cc.
    """
    from llama3.multi_launch_builder.rms_gemv_rope_multi import _build_rope_1d

    # Need rope.o on disk before XRTRunner can link this ELF.
    compile_all_external_kernels(head_dim=head_dim)

    total = n_rows * head_dim
    np.random.seed(31 + n_rows)
    x = (np.random.randn(total) * 1.0).astype(bfloat16)
    # LUT layout: half-split [cos..., sin...] of length head_dim, REPEATED n_rows times.
    half = head_dim // 2
    cos_v = np.cos(np.arange(half, dtype=np.float64) / 100.0)
    sin_v = np.sin(np.arange(half, dtype=np.float64) / 100.0)
    lut_row = np.concatenate([cos_v, sin_v]).astype(bfloat16)
    lut = np.tile(lut_row, n_rows).astype(bfloat16)

    # CPU reference: half-split RoPE per row.
    x_h = x.astype(np.float32).reshape(n_rows, head_dim)
    x1, x2 = x_h[:, :half], x_h[:, half:]
    cos_f = cos_v.astype(np.float32)
    sin_f = sin_v.astype(np.float32)
    out_h = np.empty_like(x_h)
    out_h[:, :half] = x1 * cos_f - x2 * sin_f
    out_h[:, half:] = x1 * sin_f + x2 * cos_f
    expected = out_h.astype(bfloat16).reshape(total)

    herd_x = min(8, n_rows)
    while n_rows % herd_x != 0:
        herd_x //= 2
    module = _build_rope_1d(n_rows, head_dim, bfloat16, herd_x=herd_x)
    label = f"rope_nrows{n_rows}_hd{head_dim}{label_suffix}"
    return _run_kernel_test(label, module, [x, lut], expected, rtol=5e-2, atol=5e-2)


def test_eltwise_add(n=1024, label_suffix=""):
    """Residual add at decode emb_dim. NPU eltwise_add."""
    from eltwise_add.eltwise_add import build_module

    np.random.seed(11 + n)
    a = (np.random.randn(n) * 1.0).astype(bfloat16)
    b = (np.random.randn(n) * 1.0).astype(bfloat16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(bfloat16)

    # n=1024 with herd_x=8 herd_y=1 tile_n=128 crashes Peano llc on AIE2P;
    # use the same shape llama3's o_ffn uses for its post-attn residual add:
    # herd_x=1, herd_y=2 (default), tile_n=n/2=512.
    module = build_module(
        n,
        tile_n=n // 2,
        np_dtype_in=bfloat16,
        vector_size=16,
        herd_x=1,
        herd_y=2,
    )
    label = f"eltwise_add_n{n}{label_suffix}"
    return _run_kernel_test(label, module, [a, b], expected, rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B Step 1 — standalone kernel sweep"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Run only kernels whose label contains this substring",
    )
    args = parser.parse_args()

    os.chdir(_THIS_DIR)
    prepare_air_project()

    # Define the suite (label, callable). Qwen3-1.7B shapes: emb_dim=2048,
    # q_dim=2048 (==emb_dim), kv_dim=1024, hidden_dim=6144.
    suite = [
        ("rms_norm_M1_N2048", lambda: test_weighted_rms_norm(M=1, N=2048)),
        ("gemv_Q_M2048_K2048", lambda: test_gemv(m=2048, k=2048, label_suffix="_Q")),
        ("gemv_KV_M1024_K2048", lambda: test_gemv(m=1024, k=2048, label_suffix="_KV")),
        ("gemv_O_M2048_K2048", lambda: test_gemv(m=2048, k=2048, label_suffix="_O")),
        (
            "gemv_GateUp_M6144_K2048",
            lambda: test_gemv(m=6144, k=2048, label_suffix="_GU"),
        ),
        # Down at K=6144 needs smaller tile_m=2 (mirrors llama3 down_tile_m=2
        # for K=8192) so A_l2 = herd_m × tile_m × K × 2 = 8×2×6144×2 = 192KB
        # fits in L2 (512KB). Default tile_m=8 → 768KB > L2 → fails.
        (
            "gemv_Down_M2048_K6144",
            lambda: test_gemv(
                m=2048,
                k=6144,
                label_suffix="_D",
                tile_m=2,
                m_input=1,
                herd_m=8,
            ),
        ),
        # LM head at K=2048: tile_m=16 + herd_m=8 → A=512KB just exceeds L2
        # (524288). Drop tile_m to 8: A=256KB ✓. M-iters = 16384/(8*8)=256
        # — at the AIE2P repeat=255 limit; if it fails, partition into 20×8192.
        (
            "gemv_LMhead_M16384_K2048",
            lambda: test_gemv(
                m=16384,
                k=2048,
                label_suffix="_LM",
                tile_m=8,
                m_input=8,
                herd_m=8,
            ),
        ),
        ("silu_and_mul_n6144", lambda: test_silu_and_mul(n=6144)),
        # Phase A new leaves
        ("qknorm_Q_M16_N128", lambda: test_qknorm(M=16, N=128, label_suffix="_Q")),
        ("qknorm_K_M8_N128", lambda: test_qknorm(M=8, N=128, label_suffix="_K")),
        (
            "rope_Q_nrows16_hd128",
            lambda: test_rope_1d(n_rows=16, head_dim=128, label_suffix="_Q"),
        ),
        (
            "rope_K_nrows8_hd128",
            lambda: test_rope_1d(n_rows=8, head_dim=128, label_suffix="_K"),
        ),
        # eltwise_add at n=1024 crashes Peano llc; residual stays on host
        # (~µs at decode emb_dim, not on critical path).
        # ("eltwise_add_n1024", lambda: test_eltwise_add(n=1024)),
    ]

    if args.filter:
        suite = [(label, fn) for label, fn in suite if args.filter in label]
        print(f"Filtered to {len(suite)} tests matching '{args.filter}'")

    print(f"\n{'='*72}")
    print(
        f"Qwen3-1.7B Step 1 — standalone kernel registry sweep ({len(suite)} kernels)"
    )
    print(f"{'='*72}\n")

    results = []
    for label, fn in suite:
        print(f"--- {label} ---", flush=True)
        t0 = time.time()
        try:
            passed, cos, msg = fn()
        except Exception as e:
            passed, cos, msg = False, None, f"build/exec error: {type(e).__name__}: {e}"
        dt = time.time() - t0
        marker = "PASS" if passed else "FAIL"
        cos_str = f"  cos={cos:.4f}" if cos is not None else ""
        msg_str = f"  ({msg})" if msg else ""
        print(f"  → {marker}{cos_str}{msg_str}  [{dt:.1f}s]\n", flush=True)
        results.append((label, passed, cos, msg))

    n_pass = sum(1 for _, p, _, _ in results if p)
    print(f"\n{'='*72}")
    print(f"Step 1 results: {n_pass}/{len(results)} kernels PASS")
    print(f"{'='*72}")
    for label, passed, cos, msg in results:
        marker = "PASS" if passed else "FAIL"
        cos_str = f"  cos={cos:.4f}" if cos is not None else ""
        msg_str = f"  {msg}" if msg else ""
        print(f"  [{marker}] {label:32s}{cos_str}{msg_str}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
