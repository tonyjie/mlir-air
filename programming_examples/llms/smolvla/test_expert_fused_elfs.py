# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone NPU correctness test for the three fused action-expert ELFs.

Compiles expert_rms_qkv_rope / expert_rms_q_rope / expert_o_ffn, runs each on
random data, and compares against a numpy mirror of the EXACT fused sub-graph.
This is a fusion-plumbing check, not the end-to-end oracle gate: it proves the
stitched ELF computes the math the builder intended, BEFORE anything is wired
into a denoise loop where a wiring error would be indistinguishable from
accumulated BFP16 drift.

Mirrors `test_vit_fused_elfs.py`, which is what caught the vision port's stale
`mm.o` link (cos 0.07) before it reached the end-to-end gate.

RoPE convention: the LUT is concatenated half-split `[cos... , sin...]`
(`smolvla_backbone_weights.generate_rope_lut`), matching `rope_halfsplit.cc`
AND lerobot's `apply_rope`. Note that `rms_gemms_rope_multi.__main__` carries a
DIFFERENT (interleaved) `_rope_ref` for its own local test -- do not reuse it.

Run under the NPU lock, worktree python:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_expert_fused_elfs.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from expert_fused_builders import (
    EXPERT_EMB_DIM,
    EXPERT_HEAD_DIM,
    EXPERT_HIDDEN_DIM,
    EXPERT_KV_DIM,
    EXPERT_N_HEADS,
    EXPERT_N_KV_HEADS,
    EXPERT_Q_DIM,
    EXPERT_SEQ,
    build_expert_o_ffn_module,
    build_expert_rms_q_rope_module,
    build_expert_rms_qkv_rope_module,
    expert_gemm_specs,
)
from shared.infra.cache import KernelCache, Profiler
from shared.infra.external_kernels import (
    compile_gemm_mm,
    compile_rope,
    compile_silu_and_mul,
)

SEQ = EXPERT_SEQ
EMB = EXPERT_EMB_DIM
QD = EXPERT_Q_DIM
KVD = EXPERT_KV_DIM
HID = EXPERT_HIDDEN_DIM
NH = EXPERT_N_HEADS
NKV = EXPERT_N_KV_HEADS
HD = EXPERT_HEAD_DIM
EPS = 1e-5
ROPE_BASE = 10000.0
GATE = 0.99


# ---------------------------------------------------------------------------
# numpy mirrors of the exact fused sub-graphs
# ---------------------------------------------------------------------------
def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rms_norm(x, w, eps=EPS):
    """fp32 sum-of-squares reduction, bf16 epilogue -- the kernel's own contract."""
    xf = np.asarray(x, np.float32)
    rstd = 1.0 / np.sqrt((xf**2).mean(-1, keepdims=True) + eps)
    return (xf * rstd * np.asarray(w, np.float32)).astype(bfloat16)


def gemm(a, b):
    """bf16 in, f32 accumulate, bf16 out -- the drain GEMM's contract."""
    return (np.asarray(a, np.float32) @ np.asarray(b, np.float32)).astype(bfloat16)


def silu_mul(g, u):
    gf = np.asarray(g, np.float32)
    return (gf / (1.0 + np.exp(-gf)) * np.asarray(u, np.float32)).astype(bfloat16)


def rope_lut(positions, head_dim=HD, base=ROPE_BASE):
    """Concatenated half-split LUT [cos..., sin...], gathered at `positions`."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.asarray(positions, np.float64), inv_freq)
    lut = np.empty((len(positions), head_dim), np.float64)
    lut[:, :half] = np.cos(ang)
    lut[:, half:] = np.sin(ang)
    return lut.astype(bfloat16)


def rope_apply(x_rows, lut_rows, head_dim=HD):
    """Half-split rotation, matching rope_halfsplit.cc and lerobot apply_rope."""
    half = head_dim // 2
    x = np.asarray(x_rows, np.float32)
    l = np.asarray(lut_rows, np.float32)
    c, s = l[:, :half], l[:, half:]
    x1, x2 = x[:, :half], x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * c - x2 * s
    out[:, half:] = x2 * c + x1 * s
    return out.astype(bfloat16)


def rope_2d(x_2d, lut_seq, n_h, head_dim=HD):
    """Apply RoPE to a (seq, n_h*head_dim) tensor in the kernel's seq-first row
    order: row index = seq_idx*n_h + head_idx, LUT repeated n_h times per pos."""
    seq = x_2d.shape[0]
    rows = np.asarray(x_2d).reshape(seq * n_h, head_dim)
    lut_rows = np.repeat(np.asarray(lut_seq)[:seq], n_h, axis=0)
    return rope_apply(rows, lut_rows, head_dim).reshape(seq, n_h * head_dim)


def _backend(name):
    # Same preset the backbone uses for its two fused ELFs.
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _stage_objects():
    """Compile every distinct external object the three ELFs link.

    Deduped by sym_suffix, which for the expert keys on (tile_m, tile_k_l1,
    tile_n) -- all three dims compile_gemm_mm bakes in. Keying on tile_n alone
    (as the shared helpers do) would make q/k/v (k1=48, n=80) and o/down
    (k1=32, n=80) fight over one filename.
    """
    specs = expert_gemm_specs()
    by_obj = {}
    for name, s in specs.items():
        by_obj.setdefault(s["obj"], (s, []))[1].append(name)
    for obj, (s, names) in sorted(by_obj.items()):
        print(
            f"  {obj:24s} tile_m={s['tile_m']:3d} tile_k_l1={s['tile_k_l1']:3d} "
            f"tile_n={s['tile_n']:4d}   <- {', '.join(names)}"
        )
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=obj,
        )
    print("  rope.o, silu_and_mul.o")
    compile_rope()
    compile_silu_and_mul()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_even(cache):
    """EVEN layer front half: RMSNorm + Q/K/V GEMM + RoPE Q + RoPE K."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    nw = rng.uniform(0.5, 1.5, (EMB,)).astype(bfloat16)
    wq = rng.uniform(-0.05, 0.05, (EMB, QD)).astype(bfloat16)
    wk = rng.uniform(-0.05, 0.05, (EMB, KVD)).astype(bfloat16)
    wv = rng.uniform(-0.05, 0.05, (EMB, KVD)).astype(bfloat16)
    # EVEN layers see absolute positions: the 50 action tokens sit AFTER the
    # 241-token prefix, so positions run 241..290 (padded rows keep counting).
    pos = np.arange(241, 241 + SEQ)
    lut = rope_lut(pos)

    normed = rms_norm(x, nw)
    q_ref = rope_2d(gemm(normed, wq), lut, NH)
    k_ref = rope_2d(gemm(normed, wk), lut, NKV)
    v_ref = gemm(normed, wv)

    z = lambda c: np.zeros((SEQ, c), bfloat16)
    args = [
        x,
        nw,
        z(EMB),
        wq,
        z(QD),
        wk,
        z(KVD),
        wv,
        z(KVD),
        np.repeat(lut, NH, axis=0).flatten(),
        np.repeat(lut, NKV, axis=0).flatten(),
        z(QD),
        z(KVD),
    ]
    res = cache.load_and_run(
        "expert_rms_qkv_rope",
        _backend("expert_rms_qkv_rope"),
        *args,
        output_indices=[8, 11, 12],
        bo_key="ex_even_test",
    )
    cv = cos(res[8].reshape(SEQ, KVD), v_ref)
    cq = cos(res[11].reshape(SEQ, QD), q_ref)
    ck = cos(res[12].reshape(SEQ, KVD), k_ref)
    print(f"  expert_rms_qkv_rope: cos q={cq:.5f} k={ck:.5f} v={cv:.5f}")
    return min(cq, ck, cv) >= GATE


def test_odd(cache):
    """ODD layer front half: RMSNorm + Q GEMM + RoPE Q."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    nw = rng.uniform(0.5, 1.5, (EMB,)).astype(bfloat16)
    wq = rng.uniform(-0.05, 0.05, (EMB, QD)).astype(bfloat16)
    # CROSS layers renormalise positions to start at 0 (expert_position_id -
    # min(expert_position_id) in smolvlm_with_expert.py).
    lut = rope_lut(np.arange(SEQ))

    normed = rms_norm(x, nw)
    q_ref = rope_2d(gemm(normed, wq), lut, NH)

    z = lambda c: np.zeros((SEQ, c), bfloat16)
    args = [
        x,
        nw,
        z(EMB),
        wq,
        z(QD),
        np.repeat(lut, NH, axis=0).flatten(),
        z(QD),
    ]
    res = cache.load_and_run(
        "expert_rms_q_rope",
        _backend("expert_rms_q_rope"),
        *args,
        output_indices=[6],
        bo_key="ex_odd_test",
    )
    cq = cos(res[6].reshape(SEQ, QD), q_ref)
    print(f"  expert_rms_q_rope:   cos q={cq:.5f}")
    return cq >= GATE


def test_o_ffn(cache):
    """Back half: O proj + residual + RMSNorm + gate/up + SiLU-mul + down + residual."""
    rng = np.random.default_rng(2)
    attn = rng.uniform(-1, 1, (SEQ, QD)).astype(bfloat16)
    wo = rng.uniform(-0.05, 0.05, (QD, EMB)).astype(bfloat16)
    xres = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    nw2 = rng.uniform(0.5, 1.5, (EMB,)).astype(bfloat16)
    wg = rng.uniform(-0.05, 0.05, (EMB, HID)).astype(bfloat16)
    wu = rng.uniform(-0.05, 0.05, (EMB, HID)).astype(bfloat16)
    wd = rng.uniform(-0.05, 0.05, (HID, EMB)).astype(bfloat16)

    o = gemm(attn, wo)
    x1 = (np.asarray(o, np.float32) + np.asarray(xres, np.float32)).astype(bfloat16)
    n2 = rms_norm(x1, nw2)
    sm = silu_mul(gemm(n2, wg), gemm(n2, wu))
    dn = gemm(sm, wd)
    out_ref = (np.asarray(dn, np.float32) + np.asarray(x1, np.float32)).astype(bfloat16)

    z = lambda c: np.zeros((SEQ, c), bfloat16)
    args = [
        attn,
        wo,
        z(EMB),
        xres,
        z(EMB),
        nw2,
        z(EMB),
        wg,
        z(HID),
        wu,
        z(HID),
        z(HID),
        wd,
        z(EMB),
        z(EMB),
    ]
    res = cache.load_and_run(
        "expert_o_ffn",
        _backend("expert_o_ffn"),
        *args,
        output_indices=[14],
        bo_key="ex_offn_test",
    )
    c = cos(res[14].reshape(SEQ, EMB), out_ref)
    print(f"  expert_o_ffn:        cos out={c:.5f}")
    return c >= GATE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["even", "odd", "offn", "all"], default="all")
    args = ap.parse_args()

    print("Staging external objects...")
    _stage_objects()

    cache = KernelCache("expert_fused_test_cache", verbose=False, profiler=Profiler())
    todo = ["even", "odd", "offn"] if args.which == "all" else [args.which]

    builders = {
        "even": ("expert_rms_qkv_rope", build_expert_rms_qkv_rope_module),
        "odd": ("expert_rms_q_rope", build_expert_rms_q_rope_module),
        "offn": ("expert_o_ffn", build_expert_o_ffn_module),
    }
    for w in todo:
        name, fn = builders[w]
        print(f"\nCompiling {name}...")
        cache.compile_and_cache(name, fn(), _backend(name))
    cache._save_manifest()

    print("\nRunning on NPU2:")
    tests = {"even": test_even, "odd": test_odd, "offn": test_o_ffn}
    results = {w: tests[w](cache) for w in todo}

    print()
    for w, ok in results.items():
        print(f"  {w:5s} {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print(f"PASS: all {len(results)} fused ELFs match the numpy mirror (>={GATE})")
        sys.exit(0)
    print("FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
