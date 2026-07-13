# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step-1 correctness check for the SmolVLA hybrid pipeline: the per-layer
post-RoPE K and raw V exported from the NPU backbone (run_backbone_prefill
return_kv=True) must match the real CPU model's cached K/V (oracle cpu_k/cpu_v,
captured from vlm_with_expert.forward(fill_kv_cache=True)).

This is the data the end-to-end pipeline injects into the CPU action expert's
past_key_values, so it must be correct BEFORE wiring the injection.

Runs the NPU backbone in-process (worktree python) using the oracle's
prefix_embed + prefix_position_ids, then compares against oracle cpu_k/cpu_v via
per-head cosine. Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_export_kv.py
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from smolvla_backbone_weights import (
    load_backbone_weights,
    SmolVLABackboneConfig,
    generate_rope_lut,
)
from smolvla_cpu_helpers import build_prefix_mask
from smolvla_backbone_prefill import compile_all_kernels, run_backbone_prefill
from shared.infra.cache import KernelCache, Profiler

NPU_SEQ_LEN = 256
KV_COS_THRESH = 0.98  # per-layer aggregate cosine floor for exported K/V


def cosine(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    o = np.load(_HERE / "smolvla_oracle.npz")
    pad_mask = o["prefix_pad_masks"].astype(bool)
    oracle_len = pad_mask.shape[0]
    cpu_k = o["cpu_k"]  # (16, 241, 5, 64)
    cpu_v = o["cpu_v"]
    lerobot_pos = o["prefix_position_ids"].astype(np.int64)

    cfg = SmolVLABackboneConfig()
    weights = load_backbone_weights("lerobot/smolvla_base", config=cfg)

    n_prefix = oracle_len - 1
    mask_o = build_prefix_mask(n_prefix, 1, pad_mask=pad_mask)
    mask_256 = np.full((NPU_SEQ_LEN, NPU_SEQ_LEN), -np.inf, dtype=np.float32)
    mask_256[:oracle_len, :oracle_len] = mask_o
    pad_mask_256 = np.zeros((NPU_SEQ_LEN,), dtype=bool)
    pad_mask_256[:oracle_len] = pad_mask
    positions_256 = np.clip(np.cumsum(pad_mask_256.astype(np.int64)) - 1, 0, None)

    # Confirm our reconstructed positions equal lerobot's actual ids.
    recon = positions_256[:oracle_len]
    assert np.array_equal(
        recon, lerobot_pos
    ), f"position_ids mismatch: {np.nonzero(recon != lerobot_pos)[0][:20]}"
    print(f"position_ids match lerobot's actual ids (max={recon.max()})")

    max_pos = int(positions_256.max()) + 1
    full_lut = generate_rope_lut(cfg, seq_len=max_pos, dtype=bfloat16)
    rope_lut_bf16 = full_lut[positions_256]

    x_f32 = np.zeros((NPU_SEQ_LEN, cfg.emb_dim), dtype=np.float32)
    x_f32[:oracle_len] = o["prefix_embed"]
    x_bf16 = x_f32.astype(bfloat16)

    cache = KernelCache(
        str(_HERE / "smolvla_block_kernel_cache"), verbose=False, profiler=Profiler()
    )
    compile_all_kernels(cache, cfg, NPU_SEQ_LEN, cpu_attn=False)

    _, _, kv_list = run_backbone_prefill(
        x_bf16,
        weights,
        cfg,
        cache,
        mask_256,
        positions_256,
        rope_lut_bf16,
        cpu_attn=False,
        verbose=False,
        return_kv=True,
    )

    print("\n" + "-" * 60)
    print("Per-layer exported-K/V cosine vs oracle CPU cache")
    print("-" * 60)
    k_cos, v_cos = [], []
    for i, (k_h, v_h) in enumerate(kv_list):
        ck = cosine(np.asarray(k_h[:oracle_len], np.float32), cpu_k[i])
        cv = cosine(np.asarray(v_h[:oracle_len], np.float32), cpu_v[i])
        k_cos.append(ck)
        v_cos.append(cv)
        fk = "" if ck > KV_COS_THRESH else "  <-- K BELOW"
        fv = "" if cv > KV_COS_THRESH else "  <-- V BELOW"
        print(f"  layer {i:2d}: K cos={ck:.6f}{fk}   V cos={cv:.6f}{fv}")

    kmin, vmin = min(k_cos), min(v_cos)
    print(f"\nK cos min={kmin:.6f} mean={np.mean(k_cos):.6f}")
    print(f"V cos min={vmin:.6f} mean={np.mean(v_cos):.6f}")
    assert kmin > KV_COS_THRESH, f"K export cosine {kmin} below {KV_COS_THRESH}"
    assert vmin > KV_COS_THRESH, f"V export cosine {vmin} below {KV_COS_THRESH}"
    print(f"\nPASS: exported K/V match CPU cache (all layers > {KV_COS_THRESH})")


if __name__ == "__main__":
    main()
