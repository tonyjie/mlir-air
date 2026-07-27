# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NPU-side worker for the SmolVLA VISION tower (bridged execution).

Sibling of `run_npu_backbone.py`: runs in the worktree's default `python` (has
`air`/`pyxrt`, NOT torch/lerobot), reads preprocessed camera images written by
the lerobot-venv driver (`smolvla_inference.py`), runs the 12-layer SigLIP ViT
encoder + the connector projection on NPU2, and writes back the per-image
(64, 960) connector embeddings — exactly what
`vlm_with_expert.embed_image(img)` returns on CPU.

ALL cameras in ONE invocation. SmolVLA has 3 cameras, so lerobot calls
`embed_image` 3x per inference. Spawning one bridge per camera would pay the
process spawn + safetensors weight load + ELF/XRT context setup 3x; here the
driver hands over all 3 images at once and we loop internally, so that setup is
paid once and images 2..N run with warm BOs (per-layer weights already resident
on device via `static_input_indices` + per-layer `bo_key`).

I/O contract (npz files, paths given as argv):
  argv[1] = in_path  : images (N, 3, 512, 512) f32 — ALREADY lerobot-preprocessed
                       (resize_with_pad to 512x512 then scaled to [-1, 1]).
  argv[2] = out_path : connector (N, 64, 960) f32 — RAW projection output. The
                       driver/lerobot applies the sqrt(960) scale afterwards in
                       embed_prefix; do NOT pre-apply it here.
                       Plus per-phase timings (t_*_ms) + t_image_ms (N,) so the
                       driver can separate bridge overhead from NPU compute.

Invoke under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        python3 run_npu_vision.py in.npz out.npz
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

# MUST run before numpy is imported — see bridge_common.limit_blas_threads
# (BLAS worker spin steals the CPU from this host-bound dispatch loop).
from bridge_common import limit_blas_threads, ensure_kernels, Timings  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

from vision_weights import load_vision_weights, SigLIPVisionConfig  # noqa: E402
from vision_prefill import compile_all_kernels, run_vit_encoder  # noqa: E402
from shared.infra.cache import KernelCache, Profiler  # noqa: E402

MODEL_ID = "lerobot/smolvla_base"
SEQ_LEN = 1024
CACHE_DIR = "vision_kernel_cache"
# Every ELF the fused vision path + connector needs; a cache missing any of
# these is rebuilt from scratch (see bridge_common.ensure_kernels).
EXPECTED_KERNELS = {
    "vit_ln_qkv",
    "vit_o_ffn",
    "flash_attn",
    "layer_norm",
    "gemm_connector",
}


def main():
    t = Timings()
    in_path, out_path = sys.argv[1], sys.argv[2]
    attn_mode = sys.argv[3] if len(sys.argv) > 3 else "flash"
    assert attn_mode in ("flash", "cpu"), attn_mode

    data = np.load(in_path)
    images = np.asarray(data["images"], dtype=np.float32)  # (N, 3, 512, 512)
    assert images.ndim == 4 and images.shape[1] == 3, images.shape
    n_img = images.shape[0]
    t.mark("npz_read")

    cfg = SigLIPVisionConfig()
    weights = load_vision_weights(MODEL_ID, dtype=bfloat16, config=cfg)
    t.mark("weight_load")

    cache = KernelCache(str(_HERE / CACHE_DIR), verbose=False, profiler=Profiler())
    compiled = ensure_kernels(
        cache,
        EXPECTED_KERNELS,
        lambda: compile_all_kernels(cache, cfg, SEQ_LEN, with_connector=True),
        tag="npu-vision",
    )
    t.mark("kernels")

    # First image pays XRT context load + BO alloc + the one-time static weight
    # upload; images 2..N reuse them (that is the whole point of batching the
    # cameras into one invocation). Both are reported.
    conn = np.empty((n_img, 64, cfg.connector_out), dtype=np.float32)
    per_image_ms = []
    for i in range(n_img):
        import time as _time

        t0 = _time.perf_counter()
        res = run_vit_encoder(
            images[i],
            weights,
            cfg,
            cache,
            return_per_layer=False,
            do_connector=True,
            verbose=False,
            attn_mode=attn_mode,
        )
        conn[i] = res["connector"]
        per_image_ms.append((_time.perf_counter() - t0) * 1e3)
    t.mark("encode")

    np.savez(
        out_path,
        connector=conn,
        t_image_ms=np.asarray(per_image_ms, np.float64),
        compiled=np.asarray(compiled),
        **t.as_npz_fields(),
    )
    t.report("npu-vision")
    print(
        f"[npu-vision] wrote {out_path}: connector{conn.shape} for {n_img} image(s); "
        f"per-image {[f'{v:.1f}' for v in per_image_ms]} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
