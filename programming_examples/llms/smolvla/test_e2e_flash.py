# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPERIMENT: SmolVLA end-to-end hybrid action-chunk gate, FlashAttention path.

Copy of test_e2e.py wired to attn_mode="flash" (registry FlashAttention,
non-causal, NO mask -- see smolvla_backbone_prefill.py's attn_mode="flash"
docstring for the mask-difference caveat: the real model is prefix-LM
(state token hidden from the first 240 tokens) plus 15 padding tokens; FA's
causal=False is FULLY non-causal, so it omits both the prefix mask and the
padding exclusion). This script measures how much that difference costs in
the real end-to-end action-chunk output, side by side with approach-B's
measured baseline (cosine 0.9971, nmse 0.0078 -- see test_e2e.py).

Does NOT replace test_e2e.py / verify_adapter.py (those stay on approach-B,
the production path). This is purely a measurement harness for the
FlashAttention experiment; it does not gate `make verify`.

Runs in the LEROBOT venv; the NPU subprocess is spawned by run_hybrid_forward.
Hold the NPU lock around the whole thing:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        ~/Projects/smolvla_playground/.venv/bin/python test_e2e_flash.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from smolvla_inference import (
    run_hybrid_forward,
    build_oracle_batch,
    normalized_mse,
    _fixed_noise,
    DEFAULT_MODEL,
)
from verify.comparators import regression_gate

# Reported (NOT enforced as a hard PASS/FAIL gate the way test_e2e.py's are --
# this is an experiment measurement, so we print the comparison against
# approach-B's locked baseline rather than assert against it).
APPROACH_B_COSINE = 0.9971
APPROACH_B_NMSE = 0.0078


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    o = np.load(_HERE / "smolvla_oracle.npz")
    ref = o["action_chunk"]  # (1,50,6) pure-CPU baseline, fixed zero noise
    print(f"oracle action_chunk shape {ref.shape}")

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    chunk = run_hybrid_forward(
        batch, policy=policy, noise=_fixed_noise(policy), attn_mode="flash"
    )
    print(f"hybrid (flash) action_chunk shape {chunk.shape}")

    assert chunk.shape == ref.shape, (chunk.shape, ref.shape)

    g = regression_gate(chunk, ref, cos_min=0.0, mse_max=float("inf"))
    nmse = normalized_mse(chunk, ref)

    print("\n" + "=" * 60)
    print("EXPERIMENT: FlashAttention e2e action-chunk comparison")
    print("=" * 60)
    print(f"  median per-position cosine = {g['cosine']:.6f}")
    print(f"  raw MSE                    = {g['mse']:.3e}")
    print(f"  normalized MSE             = {nmse:.5f}")
    print(f"  max abs diff               = {float(np.abs(chunk - ref).max()):.4e}")
    print("-" * 60)
    print(
        f"  approach-B (locked baseline): cosine = {APPROACH_B_COSINE}, "
        f"nmse = {APPROACH_B_NMSE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
