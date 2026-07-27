# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A3-5 Step 4 acceptance test: the SmolVLA connector (modality projection) on NPU.

Two checks, both against `vision_oracle.npz` (dumped from the real lerobot model):

  1. KERNEL-ISOLATION check: feed the ORACLE post_ln into the host pixel-shuffle
     + the `gemm_connector` ELF and compare to oracle['connector']. This isolates
     the new GEMM ELF from the encoder's accumulated BFP16 drift, so a failure
     here is unambiguously a kernel/wiring bug.
  2. END-TO-END check: run the full 12-layer NPU encoder AND the connector, and
     compare to oracle['connector']. This is the number the hybrid pipeline
     actually feeds into embed_prefix.

Run under the NPU lock (worktree python):
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 test_vit_connector.py
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
from vision_prefill import compile_all_kernels, run_vit_encoder, _run_connector
from shared.infra.cache import KernelCache, Profiler

MODEL_ID = "lerobot/smolvla_base"
SEQ_LEN = 1024
ISOLATION_THRESH = 0.99
E2E_THRESH = 0.99


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
    print("=" * 70)
    print("SmolVLA A3-5 Step 4: connector (pixel-shuffle + 64x12288x960 GEMM) on NPU")
    print("=" * 70)

    cfg = SigLIPVisionConfig()
    weights = load_vision_weights(MODEL_ID, dtype=bfloat16, config=cfg)

    o = np.load(_HERE / "vision_oracle.npz")
    pixel_values = o["pixel_values"]  # (3, 512, 512)
    post_ln_oracle = o["post_ln"]  # (1024, 768)
    conn_oracle = o["connector"]  # (64, 960) RAW (pre sqrt(960))
    conn_scaled_oracle = o["connector_scaled"]  # (64, 960)

    cache = KernelCache("vision_kernel_cache", verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, SEQ_LEN, with_connector=True)

    # -- 1. Kernel isolation: oracle post_ln -> NPU connector --
    conn_iso = _run_connector(cache, post_ln_oracle, weights.connector_w, cfg)
    c_iso = cosine(conn_iso, conn_oracle)
    p_iso = per_token_median_cos(conn_iso, conn_oracle)
    print(
        f"\n[1] ISOLATION (oracle post_ln -> NPU connector) vs oracle['connector']:"
        f"\n    cosine = {c_iso:.6f}   per-token-median = {p_iso:.6f}"
    )

    # -- 2. End to end: NPU encoder + NPU connector --
    result = run_vit_encoder(
        pixel_values,
        weights,
        cfg,
        cache,
        return_per_layer=False,
        do_connector=True,
        verbose=False,
    )
    c_e2e = cosine(result["connector"], conn_oracle)
    p_e2e = per_token_median_cos(result["connector"], conn_oracle)
    c_post = cosine(result["post_ln"], post_ln_oracle)
    print(
        f"\n[2] END-TO-END (NPU encoder + NPU connector) vs oracle['connector']:"
        f"\n    post_ln cosine   = {c_post:.6f}"
        f"\n    connector cosine = {c_e2e:.6f}   per-token-median = {p_e2e:.6f}"
    )

    # sqrt(960) scale sanity: lerobot applies it AFTER embed_image, so our raw
    # output times sqrt(960) must reproduce oracle['connector_scaled'].
    scale = float(np.sqrt(cfg.connector_out))
    c_scaled = cosine(result["connector"] * scale, conn_scaled_oracle)
    ratio = float(
        np.linalg.norm(result["connector"] * scale) / np.linalg.norm(conn_scaled_oracle)
    )
    print(
        f"\n[3] SCALE sanity: cosine(NPU*sqrt(960), oracle['connector_scaled']) = "
        f"{c_scaled:.6f}, norm ratio = {ratio:.4f}"
    )

    print("\n" + "=" * 70)
    assert c_iso > ISOLATION_THRESH, (
        f"connector ELF in isolation cosine {c_iso} <= {ISOLATION_THRESH} — this is "
        f"a kernel/wiring bug (input was the exact oracle post_ln)"
    )
    assert c_e2e > E2E_THRESH, (
        f"end-to-end connector cosine {c_e2e} <= {E2E_THRESH}; encoder post_ln "
        f"cosine was {c_post} (the encoder's own accumulated BFP16 drift)"
    )
    print(f"PASS: isolation {c_iso:.6f} and e2e {c_e2e:.6f} both > {E2E_THRESH}")


if __name__ == "__main__":
    main()
