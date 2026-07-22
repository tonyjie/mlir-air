# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone NPU correctness test for the two fused ViT ELFs (A3-6b Lever 2).

Compiles vit_ln_qkv + vit_o_ffn, runs each on random data, and compares against
a numpy reference of the EXACT fused sub-graph (affine LN + bf16-out GEMM + bias
+ residual + GELU). A cosine >= 0.99 vs that numpy reference confirms the
stitched ELF computes the intended math (this is a fusion-plumbing check, NOT the
0.945 end-to-end oracle gate — that's test_full_vit.py).

Run under the NPU lock, worktree python:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python test_vit_fused_elfs.py
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from vit_fused_builders import build_vit_ln_qkv_module, build_vit_o_ffn_module
from shared.infra.external_kernels import compile_gemm_mm
from shared.builders.gemm_builder import gemm_registry_config, disambiguate_by_tile_n
from shared.infra.cache import KernelCache, Profiler

SEQ, EMB, HID, NH, HD = 1024, 768, 3072, 12, 64


def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def ln_affine(x, g, b, eps=1e-6):
    x = x.astype(np.float32)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g.astype(np.float32) + b.astype(np.float32)


def gelu_tanh(x):
    x = x.astype(np.float32)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _drain_backend(name):
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _stage_mm():
    """Compile every distinct mm.o the two ELFs link (drain, disambiguated)."""
    o = gemm_registry_config(SEQ, EMB, EMB, "bf16", "high")
    g = gemm_registry_config(SEQ, EMB, HID, "bf16", "high")
    d = gemm_registry_config(SEQ, HID, EMB, "bf16", "high")
    o, g, d = disambiguate_by_tile_n([o, g, d])
    seen = {}
    for s in (o, g, d):
        seen[s["sym_suffix"]] = s
    for s in seen.values():
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=s["obj"],
        )


def test_ln_qkv(cache):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    g = rng.uniform(0.5, 1.5, (EMB,)).astype(bfloat16)
    b = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)
    wq = rng.uniform(-0.05, 0.05, (EMB, EMB)).astype(bfloat16)
    wk = rng.uniform(-0.05, 0.05, (EMB, EMB)).astype(bfloat16)
    wv = rng.uniform(-0.05, 0.05, (EMB, EMB)).astype(bfloat16)
    bq = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)
    bk = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)
    bv = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)

    normed = ln_affine(x, g, b)
    q_ref = normed @ wq.astype(np.float32) + bq.astype(np.float32)
    k_ref = normed @ wk.astype(np.float32) + bk.astype(np.float32)
    v_ref = normed @ wv.astype(np.float32) + bv.astype(np.float32)

    param = np.concatenate([g.astype(bfloat16), b.astype(bfloat16)]).astype(bfloat16)
    z2 = lambda: np.zeros((SEQ, EMB), bfloat16)
    args = [
        x.reshape(-1),
        param,
        z2().reshape(-1),
        wq.reshape(-1),
        z2().reshape(-1),
        wk.reshape(-1),
        z2().reshape(-1),
        wv.reshape(-1),
        z2().reshape(-1),
        bq,
        bk,
        bv,
        z2().reshape(-1),
        z2().reshape(-1),
        z2().reshape(-1),
    ]
    res = cache.load_and_run(
        "vit_ln_qkv",
        _drain_backend("vit_ln_qkv"),
        *args,
        output_indices=[12, 13, 14],
        bo_key="ln_qkv_test",
    )
    q = res[12].reshape(SEQ, EMB)
    k = res[13].reshape(SEQ, EMB)
    v = res[14].reshape(SEQ, EMB)
    cq, ck, cv = cos(q, q_ref), cos(k, k_ref), cos(v, v_ref)
    print(f"  vit_ln_qkv: cos q={cq:.5f} k={ck:.5f} v={cv:.5f}")
    return min(cq, ck, cv) >= 0.99


def test_o_ffn(cache):
    rng = np.random.default_rng(1)
    attn = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    wo = rng.uniform(-0.05, 0.05, (EMB, EMB)).astype(bfloat16)
    bo = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)
    xres = rng.uniform(-1, 1, (SEQ, EMB)).astype(bfloat16)
    g2 = rng.uniform(0.5, 1.5, (EMB,)).astype(bfloat16)
    b2 = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)
    wfc1 = rng.uniform(-0.05, 0.05, (EMB, HID)).astype(bfloat16)
    bfc1 = rng.uniform(-0.1, 0.1, (HID,)).astype(bfloat16)
    wfc2 = rng.uniform(-0.05, 0.05, (HID, EMB)).astype(bfloat16)
    bfc2 = rng.uniform(-0.1, 0.1, (EMB,)).astype(bfloat16)

    o = attn.astype(np.float32) @ wo.astype(np.float32) + bo.astype(np.float32)
    res1 = xres.astype(np.float32) + o
    normed2 = ln_affine(res1.astype(bfloat16), g2, b2)
    fc1 = normed2 @ wfc1.astype(np.float32) + bfc1.astype(np.float32)
    gel = gelu_tanh(fc1.astype(bfloat16))
    fc2 = gel.astype(bfloat16).astype(np.float32) @ wfc2.astype(
        np.float32
    ) + bfc2.astype(np.float32)
    out_ref = fc2 + res1

    ln2p = np.concatenate([g2.astype(bfloat16), b2.astype(bfloat16)]).astype(bfloat16)
    ze = lambda c: np.zeros((SEQ, c), bfloat16).reshape(-1)
    args = [
        attn.reshape(-1),
        wo.reshape(-1),
        ze(EMB),
        bo,
        ze(EMB),
        xres.reshape(-1),
        ze(EMB),
        ln2p,
        ze(EMB),
        wfc1.reshape(-1),
        ze(HID),
        bfc1,
        ze(HID),
        ze(HID),
        wfc2.reshape(-1),
        ze(EMB),
        bfc2,
        ze(EMB),
        ze(EMB),
    ]
    res = cache.load_and_run(
        "vit_o_ffn",
        _drain_backend("vit_o_ffn"),
        *args,
        output_indices=[18],
        bo_key="o_ffn_test",
    )
    out = res[18].reshape(SEQ, EMB)
    c = cos(out, out_ref)
    print(f"  vit_o_ffn: cos out={c:.5f}")
    return c >= 0.99


def main():
    print("Staging mm.o objects...")
    _stage_mm()
    cache = KernelCache("vit_fused_test_cache", verbose=False, profiler=Profiler())
    print("Compiling vit_ln_qkv...")
    cache.compile_and_cache(
        "vit_ln_qkv",
        build_vit_ln_qkv_module(SEQ, EMB, NH, HD),
        _drain_backend("vit_ln_qkv"),
    )
    print("Compiling vit_o_ffn...")
    cache.compile_and_cache(
        "vit_o_ffn", build_vit_o_ffn_module(SEQ, EMB, HID), _drain_backend("vit_o_ffn")
    )
    cache._save_manifest()

    ok1 = test_ln_qkv(cache)
    ok2 = test_o_ffn(cache)
    if ok1 and ok2:
        print("PASS: both fused ELFs match numpy reference (>=0.99)")
        sys.exit(0)
    print("FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
