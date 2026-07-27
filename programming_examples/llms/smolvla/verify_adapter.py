# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the SmolVLA backbone hybrid port (regression gate).

SmolVLA is a continuous-output (flow-matching action-chunk) model, NOT a
token-generation model, so it does NOT use the shared verify subsystem's
token-set gate (`compute_topk_set_check` / verify_runner.py's HfRunner harness
are for autoregressive LLMs). Instead this adapter drives the end-to-end hybrid
pipeline (`smolvla_inference.run_hybrid_forward`: CPU prefix -> NPU backbone ->
CPU expert 10-step denoise) and applies the continuous-output
`regression_gate` from `verify.comparators` to the (1,50,6) action chunk vs the
pure-CPU baseline (`smolvla_oracle.npz['action_chunk']`, fixed zero noise).

Must run in the LEROBOT venv (`~/Projects/smolvla_playground/.venv/bin/python`);
the NPU backbone runs in the worktree python via the subprocess bridge inside
run_hybrid_forward. `make verify` wraps this under the NPU lock.

Entry points (structural parity with sibling verify_adapter.py, adapted to
regression):
  DEFAULT_MODEL   : HF id of the policy checkpoint.
  build_config()  : config/metadata dict.
  run_gate(...)   : run the pipeline + apply regression_gate; returns the gate
                    dict; used by `main()` (make verify) as the PASS/FAIL gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_LLMS_DIR = _HERE.parent
if str(_LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMS_DIR))

from smolvla_inference import (  # noqa: E402
    run_hybrid_forward,
    build_oracle_batch,
    normalized_mse,
    _fixed_noise,
    build_config as _inference_config,
    DEFAULT_MODEL,
)
from verify.comparators import regression_gate  # noqa: E402

# Gate thresholds, locked from the measured clean run (see Task 3.3 report):
# median per-position action-chunk cosine 0.9971 vs the pure-CPU baseline, and
# a NORMALIZED MSE of ~0.0078 (raw MSE 9.13e-4 / baseline power 0.117). Cosine
# is the strong primary gate (0.99, comfortable margin). The MSE criterion is a
# normalized MSE so the gate is magnitude-invariant (a raw absolute MSE_MAX
# would drift PASS/FAIL with action scale). NMSE_MAX 0.04 = ~5x the observed
# 0.0078, a round number with real safety margin.
COS_MIN = 0.99
NMSE_MAX = 0.04


def build_config(npu_vision: bool = False) -> dict:
    cfg = _inference_config(npu_vision)
    cfg.update({"gate": "regression", "cos_min": COS_MIN, "nmse_max": NMSE_MAX})
    return cfg


def run_gate(
    cos_min: float = COS_MIN,
    nmse_max: float = NMSE_MAX,
    npu_vision: bool = False,
) -> dict:
    """Run the hybrid pipeline once and return the gate dict. Gates on cosine
    (primary) AND magnitude-invariant normalized MSE; raw MSE kept for report.

    npu_vision: also run the SigLIP vision tower + connector on NPU (the
    `make verify-npu-vision` gate). The baseline is the SAME pure-CPU lerobot
    action chunk either way, so this measures the FULL NPU error budget
    (vision + backbone), not just the backbone's."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    o = np.load(_HERE / "smolvla_oracle.npz")
    ref = o["action_chunk"]  # (1,50,6) pure-CPU baseline (fixed zero noise)

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    chunk = run_hybrid_forward(
        batch, policy=policy, noise=_fixed_noise(policy), npu_vision=npu_vision
    )
    assert chunk.shape == ref.shape, (chunk.shape, ref.shape)

    g = regression_gate(chunk, ref, cos_min=cos_min, mse_max=float("inf"))
    g["nmse"] = normalized_mse(chunk, ref)
    g["nmse_max"] = nmse_max
    g["max_abs"] = float(np.abs(chunk - ref).max())
    g["passed"] = bool(g["cosine"] >= cos_min and g["nmse"] <= nmse_max)
    return g


def main() -> int:
    npu_vision = "--npu-vision" in sys.argv
    g = run_gate(npu_vision=npu_vision)
    stages = "NPU vision + NPU backbone" if npu_vision else "NPU backbone"
    print("=" * 60)
    print(f"SmolVLA verify: e2e action-chunk regression gate [{stages}]")
    print("=" * 60)
    for k in ("cosine", "cos_min", "mse", "nmse", "nmse_max", "max_abs", "passed"):
        print(f"  {k:8s} = {g[k]}")
    print("=" * 60)
    print("PASS" if g["passed"] else "FAIL")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
