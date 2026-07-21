# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A3-5 Step 2-3 acceptance test: the FULL 12-layer SmolVLA vision encoder
(SigLIP ViT) assembled from NPU kernels vs the real-lerobot oracle.

Two checks (mirror test_full_backbone.py):
  1. Per-layer DIAGNOSIS: for each layer i, cosine(NPU per_layer[i],
     oracle layer_hidden[i]); print all 12; assert each > 0.98.
  2. FINAL GATE: cosine(NPU post_ln, oracle post_ln) > 0.99.

Run under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_full_vit.py
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

from vision_weights import load_vision_weights, SigLIPVisionConfig
from vision_prefill import compile_all_kernels, run_vit_encoder
from shared.infra.cache import KernelCache, Profiler

MODEL_ID = "lerobot/smolvla_base"
SEQ_LEN = 1024
PER_LAYER_THRESH = 0.98
FINAL_THRESH = 0.99


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def per_token_median_cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.median(num / den))


def main():
    attn_mode = "cpu" if "--cpu-attn" in sys.argv else "flash"
    print("=" * 70)
    print(
        f"SmolVLA A3-5: FULL 12-layer NPU vision encoder (SigLIP ViT) "
        f"[attn={attn_mode}]"
    )
    print("=" * 70)

    cfg = SigLIPVisionConfig()
    weights = load_vision_weights(MODEL_ID, dtype=bfloat16, config=cfg)

    oracle_path = _HERE / "vision_oracle.npz"
    o = np.load(oracle_path)
    pixel_values = o["pixel_values"]  # (3, 512, 512)
    layer_hidden = o["layer_hidden"]  # (12, 1024, 768)
    post_ln_oracle = o["post_ln"]  # (1024, 768)
    assert layer_hidden.shape[0] == cfg.n_layers, (layer_hidden.shape, cfg.n_layers)

    # -- Compile kernels & run the full encoder on NPU --
    cache = KernelCache("vision_kernel_cache", verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, SEQ_LEN)

    result = run_vit_encoder(
        pixel_values,
        weights,
        cfg,
        cache,
        return_per_layer=True,
        do_connector=False,
        verbose=True,
        attn_mode=attn_mode,
    )
    per_layer = result["layer_hidden"]
    post_ln = result["post_ln"]
    assert len(per_layer) == cfg.n_layers, len(per_layer)

    # -- Per-layer diagnosis --
    print("\n" + "-" * 70)
    print("Per-layer cosine (NPU per_layer[i] vs oracle layer_hidden[i])")
    print("-" * 70)
    per_layer_cos = []
    for i in range(cfg.n_layers):
        c = cosine(per_layer[i], layer_hidden[i])
        ptm = per_token_median_cos(per_layer[i], layer_hidden[i])
        per_layer_cos.append(c)
        flag = "" if c > PER_LAYER_THRESH else "  <-- BELOW THRESHOLD"
        print(f"  layer {i:2d}: cosine = {c:.6f}  per-tok-med = {ptm:.6f}{flag}")

    # -- Final gate --
    cos_final = cosine(post_ln, post_ln_oracle)
    ptm_final = per_token_median_cos(post_ln, post_ln_oracle)
    print("\n" + "=" * 70)
    print(
        f"FINAL GATE: cosine(NPU post_ln, oracle post_ln) = {cos_final:.6f}  "
        f"(per-tok-med = {ptm_final:.6f})"
    )
    print("=" * 70)

    # -- Assertions (root-cause, don't loosen) --
    failed = [(i, c) for i, c in enumerate(per_layer_cos) if not c > PER_LAYER_THRESH]
    assert not failed, (
        f"Per-layer cosine below {PER_LAYER_THRESH} at layers "
        f"{[i for i, _ in failed]}: {[f'{c:.4f}' for _, c in failed]}. "
        f"The per-layer print localizes the divergence."
    )
    assert cos_final > FINAL_THRESH, (
        f"post_ln cosine {cos_final} <= {FINAL_THRESH}. Per-layer cosines: "
        f"{[f'{c:.4f}' for c in per_layer_cos]}"
    )
    print(
        f"\nPASS: all {cfg.n_layers} per-layer cosines > {PER_LAYER_THRESH}; "
        f"final gate > {FINAL_THRESH}"
    )


if __name__ == "__main__":
    main()
