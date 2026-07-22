# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark: NPU SigLIP ViT vision-encoder latency vs CPU (same 12-layer math).

Apples-to-apples INFERENCE latency for the SmolVLA vision encoder (SigLIP ViT,
12 layers, seq=1024, emb=768):
- compiles NPU kernels ONCE (excluded from timing)
- warms up (first run pays BO-alloc / weight-upload), then times N NPU
  run_vit_encoder calls (median)
- times N cpu_vit_forward calls (pure-numpy, the verified reference)

Feeds a precomputed patch_embed (1024,768) so the host im2col is excluded from
the per-inference number (im2col is a one-time patch-embed, not per-layer work);
the NPU driver accepts the (1024,768) patch_embed directly.

Baseline to beat (unfused, tag smolvla-vision-unfused-v1): NPU 281 ms, CPU 719 ms.

Run under the NPU lock, worktree python:
  flock -x -w 1800 /tmp/mlir-air-npu.lock python bench_vision.py --iters 10
"""

import argparse
import time

import numpy as np
from ml_dtypes import bfloat16

from vision_weights import load_vision_weights, SigLIPVisionConfig
from vision_prefill import compile_all_kernels, run_vit_encoder
from vision_cpu_helpers import cpu_vit_layer, layer_norm
from shared.infra.cache import KernelCache, Profiler

MODEL_ID = "lerobot/smolvla_base"
SEQ_LEN = 1024


def cpu_vit_encoder_from_patch(patch_embed, weights, cfg):
    """Same 12-layer encoder the NPU driver runs, starting from a precomputed
    patch_embed (im2col excluded, matching the NPU bench's input) — for an
    apples-to-apples CPU reference of the encoder heavy ops only."""
    x = np.asarray(patch_embed, dtype=np.float32)
    for lw in weights.layers:
        x = cpu_vit_layer(x, lw, cfg)
    return layer_norm(x, weights.post_ln_w, weights.post_ln_b, cfg.layer_norm_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--attn",
        choices=["flash", "cpu"],
        default="flash",
        help="flash = non-causal FlashAttention ELF on NPU (production); "
        "cpu = mha_bidirectional on host (diagnostic)",
    )
    ap.add_argument(
        "--skip-cpu", action="store_true", help="skip the CPU reference timing"
    )
    ap.add_argument("--cache-dir", default="bench_vision_kernel_cache")
    args = ap.parse_args()

    cfg = SigLIPVisionConfig()
    weights = load_vision_weights(MODEL_ID, dtype=bfloat16, config=cfg)

    # Use the oracle patch_embed fixture if present, else random (timing is
    # data-independent for a fixed shape).
    try:
        o = np.load("vision_oracle.npz")
        patch_embed = o["patch_embed"].astype(np.float32)  # (1024, 768)
    except FileNotFoundError:
        patch_embed = (
            np.random.default_rng(0)
            .standard_normal((SEQ_LEN, cfg.emb_dim))
            .astype(np.float32)
        )

    # ---- NPU: compile ONCE (excluded from timing) ----
    print("[bench] compiling NPU kernels (one-time, excluded from timing)...")
    t_compile0 = time.perf_counter()
    cache = KernelCache(args.cache_dir, verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, SEQ_LEN)
    t_compile = time.perf_counter() - t_compile0
    print(f"[bench] compile took {t_compile:.1f}s (one-time)")

    # warmup (first run pays BO-alloc + one-time static weight upload)
    run_vit_encoder(
        patch_embed,
        weights,
        cfg,
        cache,
        return_per_layer=False,
        do_connector=False,
        verbose=False,
        attn_mode=args.attn,
    )

    npu_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        run_vit_encoder(
            patch_embed,
            weights,
            cfg,
            cache,
            return_per_layer=False,
            do_connector=False,
            verbose=False,
            attn_mode=args.attn,
        )
        npu_times.append(time.perf_counter() - t0)
    npu = np.array(npu_times)

    cpu = None
    if not args.skip_cpu:
        # ---- CPU: same 12-layer encoder (pure numpy) ----
        cpu_vit_encoder_from_patch(patch_embed, weights, cfg)  # warmup
        cpu_times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            cpu_vit_encoder_from_patch(patch_embed, weights, cfg)
            cpu_times.append(time.perf_counter() - t0)
        cpu = np.array(cpu_times)

    print(
        f"\n=== SmolVLA vision encoder inference latency "
        f"(12 layers, seq={SEQ_LEN}, attn={args.attn}) ==="
    )
    print(f"  iters = {args.iters}")
    print(
        f"  NPU vision:  median {np.median(npu)*1e3:8.1f} ms  "
        f"[min {npu.min()*1e3:.1f}, max {npu.max()*1e3:.1f}]"
    )
    if cpu is not None:
        print(
            f"  CPU vision:  median {np.median(cpu)*1e3:8.1f} ms  "
            f"[min {cpu.min()*1e3:.1f}, max {cpu.max()*1e3:.1f}]"
        )
        ratio = np.median(npu) / np.median(cpu)
        print(
            f"  ratio NPU/CPU = {ratio:.2f}x  "
            f"({'NPU slower' if ratio > 1 else 'NPU faster'})"
        )
    print(f"  baseline (unfused) = 281 ms NPU / 719 ms CPU")
    print(f"\n  one-time NPU compile = {t_compile:.1f}s (NOT in per-inference numbers)")


if __name__ == "__main__":
    main()
