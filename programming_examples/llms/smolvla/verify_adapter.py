# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the SmolVLA backbone hybrid port (regression gate).

SmolVLA is a continuous-output (flow-matching action-chunk) model, NOT a
token-generation model, so it does NOT use the shared verify subsystem's
token-set gate (`compute_topk_set_check` / verify_runner.py's HfRunner harness
are for autoregressive LLMs). Instead this adapter drives the end-to-end
pipeline (`smolvla_inference.run_hybrid_forward`) and applies the
continuous-output `regression_gate` from `verify.comparators` to the (1,50,6)
action chunk vs the pure-CPU baseline (`smolvla_oracle.npz['action_chunk']`,
fixed zero noise).

`make verify` gates the PRODUCTION config: NPU SigLIP vision + connector,
lerobot's CPU backbone, SINGLE PROCESS. The vision encoder is then the only
NPU-induced deviation from the baseline. `make verify-npu-backbone` adds the
NPU 16-layer prefill (the fuller error budget, kept for the record).

Runs in the LEROBOT venv (`~/Projects/smolvla_playground/.venv/bin/python`),
which — with the mlir-air env sourced — also has air/pyxrt, so no subprocess is
involved. `make verify` wraps this under the NPU lock.

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
    warmup_npu,
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


def build_config(
    npu_vision: bool = True,
    npu_backbone: bool = False,
    npu_expert: bool = False,
    bridge: bool = False,
) -> dict:
    cfg = _inference_config(npu_vision, npu_backbone, npu_expert, bridge)
    cfg.update({"gate": "regression", "cos_min": COS_MIN, "nmse_max": NMSE_MAX})
    return cfg


def run_gate(
    cos_min: float = COS_MIN,
    nmse_max: float = NMSE_MAX,
    npu_vision: bool = True,
    npu_backbone: bool = False,
    npu_expert: bool = False,
    bridge: bool = False,
) -> dict:
    """Run the pipeline once and return the gate dict. Gates on cosine
    (primary) AND magnitude-invariant normalized MSE; raw MSE kept for report.

    The baseline is the SAME pure-CPU lerobot action chunk for every config, so
    the gate always measures the FULL NPU error budget of whatever stages are
    enabled. Defaults = the production config (NPU vision + CPU backbone,
    single process), whose only NPU-induced deviation is the vision encoder."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    o = np.load(_HERE / "smolvla_oracle.npz")
    ref = o["action_chunk"]  # (1,50,6) pure-CPU baseline (fixed zero noise)

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    if not bridge:
        warmup_npu(npu_vision=npu_vision, npu_backbone=npu_backbone)
    chunk = run_hybrid_forward(
        batch,
        policy=policy,
        noise=_fixed_noise(policy),
        npu_vision=npu_vision,
        npu_backbone=npu_backbone,
        npu_expert=npu_expert,
        bridge=bridge,
    )
    assert chunk.shape == ref.shape, (chunk.shape, ref.shape)

    g = regression_gate(chunk, ref, cos_min=cos_min, mse_max=float("inf"))
    g["nmse"] = normalized_mse(chunk, ref)
    g["nmse_max"] = nmse_max
    g["max_abs"] = float(np.abs(chunk - ref).max())
    g["passed"] = bool(g["cosine"] >= cos_min and g["nmse"] <= nmse_max)
    return g


def main() -> int:
    """Default gate = the production config (NPU vision, CPU backbone, single
    process). Flags select the other configs:
      --npu-backbone  also run the 16-layer prefill on NPU
      --npu-expert    also run the action expert's 16 layers x 10 steps on NPU
      --cpu-vision    keep lerobot's CPU vision tower
      --bridge        drive the NPU stages through the legacy subprocesses"""
    npu_vision = "--cpu-vision" not in sys.argv
    npu_backbone = "--npu-backbone" in sys.argv
    npu_expert = "--npu-expert" in sys.argv
    bridge = "--bridge" in sys.argv
    g = run_gate(
        npu_vision=npu_vision,
        npu_backbone=npu_backbone,
        npu_expert=npu_expert,
        bridge=bridge,
    )
    cfg = build_config(npu_vision, npu_backbone, npu_expert, bridge)
    print("=" * 60)
    print("SmolVLA verify: e2e action-chunk regression gate")
    print(f"  NPU stages     : {cfg['npu_stages']}")
    print(f"  execution model: {cfg['execution_model']}")
    print("=" * 60)
    for k in ("cosine", "cos_min", "mse", "nmse", "nmse_max", "max_abs", "passed"):
        print(f"  {k:8s} = {g[k]}")
    print("=" * 60)
    print("PASS" if g["passed"] else "FAIL")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
