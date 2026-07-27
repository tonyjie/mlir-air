# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-ELF device breakdown of the SHIPPING NPU vision encoder (A3-8 study).

Runs the deployed `VisionRuntime.encode` path (NPU vision, 3 images, fused
3-ELF/layer config) with `shared.infra.cache.Profiler` enabled, so every
`load_and_run` records BO Write / NPU Run / BO Read. Prints:

  (1) the per-ELF fine-grained breakdown (Profiler.report)
  (2) the wall-time reconciliation: encode wall vs sum of driver calls vs
      sum of device time, and the host residual.

Nothing here changes the deployment; it only flips the profiler flag on the
runtime's KernelCache after warmup.

  flock -x -w 1800 /tmp/mlir-air-npu.lock python3 scripts/vision_profile_run.py
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
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5, help="timed encode() repeats")
    ap.add_argument("--out", default="results/vision_profile.json")
    args = ap.parse_args()

    from smolvla_npu_runtime import VisionRuntime

    rt = VisionRuntime(verbose=False)
    print(f"[setup] {rt.setup_ms}")

    rng = np.random.default_rng(0)
    imgs = [
        rng.uniform(-1, 1, (3, 512, 512)).astype(np.float32) for _ in range(args.images)
    ]

    # Warmup: pays XRT ctx + BO alloc + one-time static weight upload.
    rt.encode(imgs)
    rt.encode(imgs)

    # ---- timed, profiler ON ----
    rt.cache.profiler.enabled = True
    walls, im2cols, encodes, per_img = [], [], [], []
    for _ in range(args.reps):
        t = {}
        t0 = time.perf_counter()
        rt.encode(imgs, timings=t)
        walls.append((time.perf_counter() - t0) * 1e3)
        im2cols.append(t["vision"]["t_im2col_ms"])
        encodes.append(t["vision"]["t_encode_ms"])
        per_img.append(t["vision"]["t_image_ms"])

    rt.cache.profiler.report()

    prof = rt.cache.profiler
    n_rep = args.reps
    rows = {}
    for name, entries in prof.kernel_breakdowns.items():
        n = len(entries)
        rows[name] = {
            "invocations_total": n,
            "invocations_per_image": n / (n_rep * args.images),
            "write_ms": sum(e["write_ms"] for e in entries) / n,
            "kernel_ms": sum(e["kernel_ms"] for e in entries) / n,
            "read_ms": sum(e["read_ms"] for e in entries) / n,
        }
    # Driver wall per call (includes filelock acquire + xrt.run construction).
    for name, times in prof.kernel_times.items():
        rows[name]["driver_ms"] = sum(times) / len(times) * 1e3

    tot_dev = sum(r["kernel_ms"] * r["invocations_total"] for r in rows.values())
    tot_wr = sum(r["write_ms"] * r["invocations_total"] for r in rows.values())
    tot_rd = sum(r["read_ms"] * r["invocations_total"] for r in rows.values())
    tot_drv = sum(r["driver_ms"] * r["invocations_total"] for r in rows.values())

    print("\n=== A3-8 per-ELF summary (per IMAGE, averaged) ===")
    print(
        f"  {'ELF':16s} {'n/img':>6s} {'write':>9s} {'NPU run':>9s} {'read':>8s} "
        f"{'driver':>9s} {'dev/img':>9s} {'drv/img':>9s}"
    )
    n_img_total = n_rep * args.images
    dev_img = drv_img = 0.0
    for name in sorted(
        rows, key=lambda k: -rows[k]["kernel_ms"] * rows[k]["invocations_total"]
    ):
        r = rows[name]
        npi = r["invocations_per_image"]
        d = r["kernel_ms"] * npi
        v = r["driver_ms"] * npi
        dev_img += d
        drv_img += v
        print(
            f"  {name:16s} {npi:6.1f} {r['write_ms']:8.3f}ms {r['kernel_ms']:8.3f}ms "
            f"{r['read_ms']:7.3f}ms {r['driver_ms']:8.3f}ms {d:8.2f}ms {v:8.2f}ms"
        )
    print(
        f"  {'TOTAL':16s} {'':6s} {'':9s} {'':9s} {'':8s} {'':9s} {dev_img:8.2f}ms {drv_img:8.2f}ms"
    )

    wall_med = float(np.median(walls))
    enc_med = float(np.median(encodes))
    im_med = float(np.median(im2cols))
    print("\n=== wall reconciliation (per encode() call, median of reps) ===")
    print(f"  encode() wall            {wall_med:8.2f} ms   ({args.images} images)")
    print(f"    im2col+to_chw (host)   {im_med:8.2f} ms")
    print(f"    encode loop            {enc_med:8.2f} ms")
    print(f"      driver calls (sum)   {drv_img*args.images:8.2f} ms")
    print(f"        BO write           {tot_wr/n_rep:8.2f} ms")
    print(f"        NPU run            {tot_dev/n_rep:8.2f} ms")
    print(f"        BO read            {tot_rd/n_rep:8.2f} ms")
    print(
        f"        lock+xrt.run glue  "
        f"{(tot_drv - tot_dev - tot_wr - tot_rd)/n_rep:8.2f} ms"
    )
    print(
        f"      host glue in loop      "
        f"{enc_med - drv_img*args.images:8.2f} ms   (residual)"
    )
    print(
        f"  per-image walls          {[f'{v:.1f}' for v in per_img[len(per_img)//2]]}"
    )

    out = {
        "rows": rows,
        "images": args.images,
        "reps": args.reps,
        "wall_ms_median": wall_med,
        "im2col_ms_median": im_med,
        "encode_loop_ms_median": enc_med,
        "walls": walls,
        "encodes": encodes,
        "im2cols": im2cols,
        "per_image_ms": per_img,
        "device_ms_per_image": dev_img,
        "driver_ms_per_image": drv_img,
    }
    p = _HERE / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
