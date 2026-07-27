# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Why is image 1 of every encode() ~38 ms slower than images 2 and 3?

Measured per-image walls inside one `VisionRuntime.encode([img,img,img])`:
    [176.2, 138.6, 133.4] [173.5, 133.5, 133.2] [176.8, 139.9, 138.5] ...
Hypothesis: `encode` runs the im2col patch-embed for all 3 images with ALL
threads *before* entering `npu_thread_limits(1)` (deliberately: 4.2 ms
multi-threaded vs 9.5 ms at 1 thread). The OpenBLAS workers then busy-spin for
a while and preempt the NPU dispatch thread during image 1.

A/B, interleaved round-robin (process-to-process drift is larger than the
effect):
  A = shipping behaviour (im2col multi-threaded, outside the clamp)
  B = whole encode inside npu_thread_limits(1) (im2col single-threaded too)

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_first_image_ab.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default="results/vision_first_image_ab.json")
    args = ap.parse_args()

    from smolvla_npu_runtime import VisionRuntime, npu_thread_limits

    rt = VisionRuntime(verbose=False)
    rng = np.random.default_rng(0)
    imgs = [rng.uniform(-1, 1, (3, 512, 512)).astype(np.float32) for _ in range(3)]
    rt.encode(imgs)
    rt.encode(imgs)

    res = {"A_shipping": [], "B_clamped_im2col": []}
    per_img = {"A_shipping": [], "B_clamped_im2col": []}
    for _ in range(args.rounds):
        for arm in ("A_shipping", "B_clamped_im2col"):
            t = {}
            t0 = time.perf_counter()
            if arm == "A_shipping":
                rt.encode(imgs, timings=t)
            else:
                with npu_thread_limits(1):
                    rt.encode(imgs, timings=t)
            res[arm].append((time.perf_counter() - t0) * 1e3)
            per_img[arm].append(t["vision"]["t_image_ms"])

    print(f"\n=== first-image A/B ({args.rounds} interleaved rounds) ===")
    for arm in res:
        w = np.median(res[arm])
        pi = np.median(np.array(per_img[arm]), axis=0)
        print(
            f"  {arm:18s} wall {w:7.1f} ms   per-image "
            f"[{pi[0]:6.1f} {pi[1]:6.1f} {pi[2]:6.1f}]  "
            f"img1 penalty {pi[0]-np.mean(pi[1:]):+6.1f} ms"
        )
    a, b = np.median(res["A_shipping"]), np.median(res["B_clamped_im2col"])
    print(f"  B vs A: {a-b:+.1f} ms ({a/b:.3f}x)")

    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {k: {"walls": res[k], "per_image": per_img[k]} for k in res}, indent=2
        )
    )
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
