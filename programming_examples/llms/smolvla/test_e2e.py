# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end hybrid action-chunk gate (Task 3.3).

Compares run_hybrid_forward's (1,50,6) action chunk (CPU prefix -> NPU backbone
-> CPU expert 10-step denoise) against the pure-CPU oracle baseline
(smolvla_oracle.npz['action_chunk'], same fixed zero noise) via regression_gate.

Runs in the LEROBOT venv; the NPU subprocess is spawned by run_hybrid_forward.
Hold the NPU lock around the whole thing:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        ~/Projects/smolvla_playground/.venv/bin/python test_e2e.py

Threshold rationale (see Task 3.3 report): thresholds are set from the observed
clean-run distribution, not blindly inherited. The exported K/V match the CPU
cache at cos>=0.995 (test_export_kv); the residual e2e action-chunk gap is bf16
backbone noise propagated through the 10-step flow-matching integrator.
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
    _fixed_noise,
    DEFAULT_MODEL,
)
from verify.comparators import regression_gate

# Thresholds locked from the measured clean run (see report). Action-chunk
# median per-position cosine and MSE against the pure-CPU baseline.
COS_MIN = 0.99
MSE_MAX = 1e-3


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    o = np.load(_HERE / "smolvla_oracle.npz")
    ref = o["action_chunk"]  # (1,50,6) pure-CPU baseline, fixed zero noise
    print(f"oracle action_chunk shape {ref.shape}")

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    chunk = run_hybrid_forward(batch, policy=policy, noise=_fixed_noise(policy))
    print(f"hybrid action_chunk shape {chunk.shape}")

    assert chunk.shape == ref.shape, (chunk.shape, ref.shape)

    g = regression_gate(chunk, ref, cos_min=COS_MIN, mse_max=MSE_MAX)
    print("\n" + "=" * 60)
    print("E2E ACTION-CHUNK GATE (hybrid vs pure-CPU baseline)")
    print("=" * 60)
    print(f"  median per-position cosine = {g['cosine']:.6f}  (min {COS_MIN})")
    print(f"  MSE                        = {g['mse']:.3e}  (max {MSE_MAX:.1e})")
    print(f"  max abs diff               = {float(np.abs(chunk - ref).max()):.4e}")
    print(f"  passed                     = {g['passed']}")
    print("=" * 60)

    assert g["passed"], f"e2e action gate FAIL: {g}"
    print("\nPASS: hybrid action chunk matches the pure-CPU baseline within gate.")


if __name__ == "__main__":
    main()
