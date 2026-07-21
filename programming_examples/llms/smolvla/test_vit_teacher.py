# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Teacher-forced per-layer diagnostic: feed each NPU encoder block the ORACLE
input of that layer (layer_hidden[i-1], or patch_embed for L0) and measure its
output vs oracle layer_hidden[i], independently. Decouples per-layer kernel
error from accumulation drift.
"""

import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from vision_weights import load_vision_weights, SigLIPVisionConfig
from vision_prefill import compile_all_kernels, run_vit_block
from shared.infra.cache import KernelCache, Profiler


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    attn_mode = "cpu" if "--cpu-attn" in sys.argv else "flash"
    cfg = SigLIPVisionConfig()
    w = load_vision_weights("lerobot/smolvla_base", dtype=bfloat16, config=cfg)
    o = np.load(_HERE / "vision_oracle.npz")
    pe = o["patch_embed"]
    LH = o["layer_hidden"]
    cache = KernelCache("vision_kernel_cache", verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, 1024)
    print(f"\nTeacher-forced per-layer cosine [attn={attn_mode}]:")
    for i, lw in enumerate(w.layers):
        xin = (pe if i == 0 else LH[i - 1]).astype(bfloat16)
        out = run_vit_block(xin, lw, cfg, cache, layer_idx=i, attn_mode=attn_mode)
        print(f"  layer {i:2d}: cos(out, oracle[{i}]) = {cosine(out, LH[i]):.6f}")


if __name__ == "__main__":
    main()
