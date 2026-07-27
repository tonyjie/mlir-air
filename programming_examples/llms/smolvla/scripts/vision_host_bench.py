# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side accounting for the NPU vision segment (A3-8 study, deliverable §3).

Times every non-device cost the shipping `VisionRuntime.encode` path pays, at
the exact shapes and dtypes it uses, so the 465 ms wall can be reconciled
against (device time + host time + driver glue).

No NPU: pure host measurement. Run on an otherwise idle machine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

SEQ, EMB, HID = 1024, 768, 3072
N_LAYERS = 12


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/vision_host.json")
    args = ap.parse_args()

    from vision_weights import load_vision_weights, SigLIPVisionConfig
    from vision_cpu_helpers import im2col_patch_embed, pixel_shuffle
    from smolvla_npu_runtime import npu_thread_limits, _to_chw_f32

    cfg = SigLIPVisionConfig()
    w = load_vision_weights("lerobot/smolvla_base", dtype=bfloat16, config=cfg)

    rng = np.random.default_rng(0)
    img_np = rng.uniform(-1, 1, (3, 512, 512)).astype(np.float32)
    try:
        import torch

        img_t = torch.from_numpy(img_np).unsqueeze(
            0
        )  # (1,3,512,512), what lerobot passes
    except ImportError:
        img_t = None

    patch = im2col_patch_embed(
        img_np, w.patch_w, w.patch_b, w.pos_embed, cfg.patch_size
    )
    patch_bf16 = patch.astype(bfloat16)
    post_ln_bf16 = rng.standard_normal(SEQ * EMB).astype(np.float32).astype(bfloat16)
    post_ln_bf16 = post_ln_bf16.reshape(SEQ, EMB)

    r = {}

    # --- per-image, once, before the dispatch loop ---
    if img_t is not None:
        r["to_chw_f32 (torch 1x3x512x512)"] = (bench(lambda: _to_chw_f32(img_t)), 3)
    r["to_chw_f32 (numpy 3x512x512)"] = (bench(lambda: _to_chw_f32(img_np)), 3)
    r["im2col_patch_embed (all threads)"] = (
        bench(
            lambda: im2col_patch_embed(
                img_np, w.patch_w, w.patch_b, w.pos_embed, cfg.patch_size
            ),
            n=15,
        ),
        3,
    )
    with npu_thread_limits(1):
        r["im2col_patch_embed (1 thread)"] = (
            bench(
                lambda: im2col_patch_embed(
                    img_np, w.patch_w, w.patch_b, w.pos_embed, cfg.patch_size
                ),
                n=15,
            ),
            0,
        )
    r["patch_embed f32->bf16 (1024x768)"] = (
        bench(lambda: patch.astype(bfloat16)),
        3,
    )

    # --- per LAYER inside the dispatch loop (x12 per image, x3 images) ---
    x = patch_bf16
    r["np.zeros (1024,768) bf16  [FA out]"] = (
        bench(lambda: np.zeros((SEQ, EMB), dtype=bfloat16)),
        N_LAYERS * 3,
    )
    r["ascontiguousarray(bf16 (1024,768))"] = (
        bench(lambda: np.ascontiguousarray(np.asarray(x, dtype=bfloat16)).reshape(-1)),
        N_LAYERS * 3 * 6,  # ln_args[0], q,k,v to FA, offn_args[0], offn_args[5]
    )

    # --- once per image, tail ---
    r["post_ln bf16->f32 (1024x768)"] = (
        bench(lambda: np.asarray(post_ln_bf16, dtype=np.float32)),
        3,
    )
    post_ln_f32 = np.asarray(post_ln_bf16, np.float32)
    r["pixel_shuffle (1024,768)->(64,12288)"] = (
        bench(lambda: pixel_shuffle(post_ln_f32, 4)),
        3,
    )
    shuffled = pixel_shuffle(post_ln_f32, 4)
    r["connector A f32->bf16 (64,12288)"] = (
        bench(
            lambda: np.ascontiguousarray(np.asarray(shuffled, dtype=bfloat16)).reshape(
                -1
            )
        ),
        3,
    )
    r["connector B ascontiguous (12288,960) bf16"] = (
        bench(
            lambda: np.ascontiguousarray(
                np.asarray(w.connector_w, dtype=bfloat16)
            ).reshape(-1)
        ),
        3,
    )
    r["LN param np.concatenate (2x768)"] = (
        bench(
            lambda: np.concatenate(
                [
                    np.asarray(w.post_ln_w, dtype=bfloat16),
                    np.asarray(w.post_ln_b, dtype=bfloat16),
                ]
            ).astype(bfloat16)
        ),
        3,
    )
    r["np.zeros (1024,768) bf16  [LN out]"] = (
        bench(lambda: np.zeros(SEQ * EMB, dtype=bfloat16)),
        3,
    )

    # --- once per encode() ---
    def clamp_cycle():
        with npu_thread_limits(1):
            pass

    r["npu_thread_limits enter+exit"] = (bench(clamp_cycle), 1)

    print(f"\n{'host op':44s} {'each':>9s} {'x/encode':>9s} {'total':>9s}")
    print("-" * 76)
    tot = 0.0
    for k, (ms, cnt) in r.items():
        t = ms * cnt
        tot += t
        print(f"{k:44s} {ms:8.3f}ms {cnt:9d} {t:8.2f}ms")
    print("-" * 76)
    print(f"{'TOTAL host (per 3-image encode)':44s} {'':9s} {'':9s} {tot:8.2f}ms")

    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {k: {"ms_each": v[0], "count": v[1]} for k, v in r.items()}, indent=2
        )
    )
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
