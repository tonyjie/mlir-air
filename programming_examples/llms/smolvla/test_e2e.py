# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end hybrid action-chunk gate (Task 3.3).

Compares run_hybrid_forward's (1,50,6) action chunk (CPU prefix -> NPU backbone
-> CPU expert 10-step denoise) against the pure-CPU oracle baseline
(smolvla_oracle.npz['action_chunk'], same fixed zero noise) via regression_gate.

Runs in the LEROBOT venv, which also has air/pyxrt: run_hybrid_forward drives
the NPU backbone IN-PROCESS (bridge=False default; pass bridge=True for the
legacy subprocess path).
Hold the NPU lock around the whole thing:
    flock -x -w 1800 /tmp/mlir-air-npu.lock \
        ~/Projects/smolvla_playground/.venv/bin/python test_e2e.py

Threshold rationale (see Task 3.3 report): thresholds are set from the observed
clean-run distribution, not blindly inherited. The exported K/V match the CPU
cache at cos>=0.995 (test_export_kv); the residual e2e action-chunk gap is bf16
backbone noise propagated through the 10-step flow-matching integrator.

The MSE criterion is a NORMALIZED MSE (nmse = mean((chunk-ref)**2) /
mean(ref**2)) so the gate is invariant to action magnitude — a raw absolute
MSE_MAX would drift PASS/FAIL as the action chunk's scale changes across
prompts/observations without any real regression. See NMSE_MAX below for the
measured clean value and the chosen margin.
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

# Thresholds locked from the measured clean run (see report). Cosine stays the
# strong primary gate. The MSE criterion is a NORMALIZED MSE (relative to the
# baseline action power) so it is magnitude-invariant.
#   measured clean run: cosine 0.9971, nmse ~0.0078 (raw MSE 9.13e-4 /
#   baseline power mean(ref**2)=0.117).
# NMSE_MAX = 0.04 gives ~5x safety margin over the observed 0.0078 (round
# number), so a normal prompt/recompile cannot flip the gate without a real
# regression.
COS_MIN = 0.99
NMSE_MAX = 0.04


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

    # regression_gate reports raw MSE + cosine; the gate decision uses cosine
    # (primary) AND the magnitude-invariant normalized MSE.
    g = regression_gate(chunk, ref, cos_min=COS_MIN, mse_max=float("inf"))
    nmse = normalized_mse(chunk, ref)
    passed = bool(g["cosine"] >= COS_MIN and nmse <= NMSE_MAX)
    print("\n" + "=" * 60)
    print("E2E ACTION-CHUNK GATE (hybrid vs pure-CPU baseline)")
    print("=" * 60)
    print(f"  median per-position cosine = {g['cosine']:.6f}  (min {COS_MIN})")
    print(f"  raw MSE                    = {g['mse']:.3e}  (diagnostic)")
    print(f"  normalized MSE             = {nmse:.5f}  (max {NMSE_MAX})")
    print(f"  max abs diff               = {float(np.abs(chunk - ref).max()):.4e}")
    print(f"  passed                     = {passed}")
    print("=" * 60)

    assert passed, (
        f"e2e action gate FAIL: cosine={g['cosine']:.6f} (min {COS_MIN}), "
        f"nmse={nmse:.5f} (max {NMSE_MAX})"
    )
    print("\nPASS: hybrid action chunk matches the pure-CPU baseline within gate.")


if __name__ == "__main__":
    main()
