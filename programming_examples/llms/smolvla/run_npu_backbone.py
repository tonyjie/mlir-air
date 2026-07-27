# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NPU-side worker for the SmolVLA hybrid pipeline (bridged execution).

Runs in the worktree's default `python` (has `air`/`pyxrt`, NOT torch/lerobot).
Reads a prefix-embed fixture written by the lerobot-venv driver
(`smolvla_inference.py`), runs the full 16-layer backbone prefill on NPU2 with
NPU attention (cpu_attn=False), and writes back the per-layer post-RoPE K and
raw V (5 kv-heads) -- exactly the quantities the SmolVLA action expert caches
during its `fill_kv_cache=True` prefill.

I/O contract (npz files, paths given as argv):
  argv[1] = in_path   : prefix_embed (ORACLE_LEN, emb_dim) f32,
                        pad_mask (ORACLE_LEN,) bool,
                        position_ids (ORACLE_LEN,) int64  [lerobot's actual ids]
  argv[2] = out_path  : k (n_layers, ORACLE_LEN, n_kv_heads, head_dim) f32,
                        v (n_layers, ORACLE_LEN, n_kv_heads, head_dim) f32,
                        final_hidden (ORACLE_LEN, emb_dim) f32,
                        per_layer (n_layers, ORACLE_LEN, emb_dim) f32
  argv[3] = attn_mode : OPTIONAL, one of {"gemm","flash"} (default "gemm" =
                        approach-B, the production NPU attention path).
                        "flash" is the EXPERIMENT path (registry
                        FlashAttention, non-causal, no mask -- see
                        smolvla_backbone_prefill.py's attn_mode="flash"
                        docstring for the mask-difference caveat).

Invoke under the NPU lock:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        python3 run_npu_backbone.py in.npz out.npz [attn_mode]
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

# MUST run before numpy is imported — see bridge_common.limit_blas_threads.
from bridge_common import limit_blas_threads, ensure_kernels, Timings  # noqa: E402

limit_blas_threads()

import numpy as np  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

from smolvla_backbone_weights import (
    load_backbone_weights,
    SmolVLABackboneConfig,
    generate_rope_lut,
)
from smolvla_cpu_helpers import build_padded_mask_and_positions
from smolvla_backbone_prefill import compile_all_kernels, run_backbone_prefill
from shared.infra.cache import KernelCache, Profiler

NPU_SEQ_LEN = 256  # registry kernels validated at M=256

# ELFs each attention mode needs. A cache missing any of them is rebuilt
# (bridge_common.ensure_kernels); otherwise the per-inference bridge would pay
# the full aircc rebuild every call, which no deployment would do.
EXPECTED_KERNELS = {
    "gemm": {"rms_gemms_rope", "o_ffn", "qkt", "pv", "masked_softmax"},
    "flash": {"rms_gemms_rope", "o_ffn", "flash_attn"},
}


def main():
    t = Timings()
    in_path, out_path = sys.argv[1], sys.argv[2]
    attn_mode = sys.argv[3] if len(sys.argv) > 3 else "gemm"
    assert attn_mode in ("gemm", "flash"), attn_mode
    data = np.load(in_path)
    prefix_embed = data["prefix_embed"]  # (ORACLE_LEN, emb_dim) f32
    pad_mask = data["pad_mask"].astype(bool)  # (ORACLE_LEN,)
    oracle_len = prefix_embed.shape[0]
    t.mark("npz_read")

    cfg = SmolVLABackboneConfig()
    weights = load_backbone_weights("lerobot/smolvla_base", config=cfg)
    t.mark("weight_load")

    mask_256, positions_256 = build_padded_mask_and_positions(
        pad_mask, oracle_len, NPU_SEQ_LEN
    )

    # Cross-check that our reconstructed positions match lerobot's actual ids
    # over the real (non-padding) region -- a mismatch here silently corrupts
    # RoPE (task escalation step 3).
    if "position_ids" in data.files:
        lerobot_pos = data["position_ids"].astype(np.int64)
        recon = positions_256[:oracle_len]
        if not np.array_equal(recon, lerobot_pos):
            diff = np.nonzero(recon != lerobot_pos)[0]
            raise RuntimeError(
                f"reconstructed position_ids disagree with lerobot's at indices "
                f"{diff.tolist()[:20]}: recon={recon[diff][:20]} "
                f"lerobot={lerobot_pos[diff][:20]}"
            )

    max_pos = int(positions_256.max()) + 1
    full_lut = generate_rope_lut(cfg, seq_len=max_pos, dtype=bfloat16)
    rope_lut_bf16 = full_lut[positions_256]  # (256, head_dim)

    x_f32 = np.zeros((NPU_SEQ_LEN, cfg.emb_dim), dtype=np.float32)
    x_f32[:oracle_len] = prefix_embed
    x_bf16 = x_f32.astype(bfloat16)

    cache = KernelCache(
        str(_HERE / "smolvla_block_kernel_cache"), verbose=False, profiler=Profiler()
    )
    compiled = ensure_kernels(
        cache,
        EXPECTED_KERNELS[attn_mode],
        lambda: compile_all_kernels(
            cache, cfg, NPU_SEQ_LEN, cpu_attn=False, attn_mode=attn_mode
        ),
        tag="npu-backbone",
    )
    t.mark("kernels")

    final_hidden, per_layer, kv_list = run_backbone_prefill(
        x_bf16,
        weights,
        cfg,
        cache,
        mask_256,
        positions_256,
        rope_lut_bf16,
        cpu_attn=False,
        attn_mode=attn_mode,
        verbose=True,
        return_kv=True,
    )
    t.mark("prefill")

    n_layers = cfg.n_layers
    k_out = np.stack(
        [np.asarray(k[:oracle_len], np.float32) for k, _ in kv_list]
    )  # (L, oracle_len, n_kv, hd)
    v_out = np.stack([np.asarray(v[:oracle_len], np.float32) for _, v in kv_list])
    per_layer_out = np.stack(
        [np.asarray(h[:oracle_len], np.float32) for h in per_layer]
    )
    np.savez(
        out_path,
        k=k_out,
        v=v_out,
        final_hidden=np.asarray(final_hidden[:oracle_len], np.float32),
        per_layer=per_layer_out,
        compiled=np.asarray(compiled),
        **t.as_npz_fields(),
    )
    assert k_out.shape == (
        n_layers,
        oracle_len,
        cfg.n_kv_heads,
        cfg.head_dim,
    ), k_out.shape
    t.report("npu-backbone")
    print(
        f"[npu] wrote {out_path}: k{k_out.shape} v{v_out.shape} "
        f"final{final_hidden.shape}"
    )


if __name__ == "__main__":
    main()
