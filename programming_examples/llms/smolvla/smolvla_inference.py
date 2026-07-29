# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end inference with the vision encoder on NPU2.

SmolVLA is three stages: a SigLIP vision encoder, a SmolLM2-360M language
backbone, and a flow-matching action expert. All three were ported to NPU2 and
verified; **only the vision encoder ships on the NPU**, because it is the only
one measurably faster there (1.19x per image, 1.07x end to end). The other two
run lerobot's own unmodified CPU path. See README.md for the measurements.

How the splice works
--------------------
`run_hybrid_forward` wraps `policy.model.embed_prefix` and, for the duration of
that call, swaps `vlm_with_expert.embed_image` for one that serves results the
NPU already computed. Everything downstream -- the sqrt(960) scale, the pad and
attention masks, prefix assembly, the backbone, the action expert -- is
untouched lerobot code, so the comparison against the pure-CPU baseline is
honest. The wrapper is always restored in a `finally`.

All camera images are encoded in ONE call into the runtime rather than one call
per image, so the NPU's weights and ELFs are touched once per inference.

Run standalone:
    python3 smolvla_inference.py            # NPU vision (default)
    python3 smolvla_inference.py --cpu      # unmodified CPU model, for comparison
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

_HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = "lerobot/smolvla_base"
DEFAULT_PROMPT = "pick up the cube"


def normalized_mse(chunk, ref) -> float:
    """MSE relative to the baseline action's own power: mean((chunk-ref)**2) /
    mean(ref**2).

    Magnitude-invariant, so the gate does not depend on the absolute scale of a
    particular prompt's action chunk -- a raw MSE threshold would silently drift
    PASS/FAIL as action magnitude changes.
    """
    chunk = np.asarray(chunk, np.float32)
    ref = np.asarray(ref, np.float32)
    power = float(np.mean(ref**2))
    return float(np.mean((chunk - ref) ** 2) / max(power, 1e-12))


def build_config(npu_vision: bool = True) -> dict:
    """Minimal config dict for the verify adapter and for reporting."""
    return {
        "model": DEFAULT_MODEL,
        "prompt": DEFAULT_PROMPT,
        "execution_model": "single-process (air/pyxrt in the lerobot venv)",
        "npu_stages": "vision" if npu_vision else "none (pure CPU)",
    }


def build_oracle_batch(policy, prompt: str = DEFAULT_PROMPT):
    """The same synthetic batch the oracle dumper uses: zero images and state,
    tokenized prompt. Deterministic, so the gate is reproducible."""
    cfg = policy.config
    b = {}
    for k, f in cfg.input_features.items():
        b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        [prompt],
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b


def fixed_noise(policy):
    """Deterministic zero noise, matching the oracle's action_chunk baseline.

    The action expert is a flow-matching denoiser seeded from noise, so a
    reproducible gate needs the noise pinned.
    """
    return torch.zeros(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        dtype=torch.float32,
    )


def warmup_npu():
    """Build the vision runtime and run one throwaway encode.

    Without this the first *measured* inference also pays XRT context creation,
    buffer-object allocation and the one-time static weight upload -- measured
    335 ms vs 148 ms warm, per image.
    """
    from smolvla_runtime import get_vision_runtime

    get_vision_runtime().warmup()


def run_hybrid_forward(
    batch,
    policy=None,
    noise=None,
    npu_vision: bool = True,
    timings: dict | None = None,
):
    """Run one `predict_action_chunk`; return the (1, chunk, action_dim) chunk.

    npu_vision : encode the camera images with the NPU SigLIP ViT + connector
        instead of lerobot's CPU vision tower. False runs the model completely
        unmodified, which is the baseline the gate compares against.
    timings    : optional dict, filled with the NPU stage's phase timings.
    """
    if policy is None:
        policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    vwe = policy.model.vlm_with_expert
    orig_embed_prefix = policy.model.embed_prefix
    orig_embed_image = vwe.embed_image

    def _wrapped_embed_prefix(*a, **kw):
        # `images` is embed_prefix's first positional arg: a list of N camera
        # tensors. Encode them ALL up front in one call, then let the untouched
        # embed_prefix pull them one at a time through the swapped embed_image
        # -- so the sqrt(960) scale, the pad/attention masks and the prefix
        # assembly all remain lerobot's own code.
        from smolvla_runtime import get_vision_runtime

        images = kw["images"] if "images" in kw else a[0]
        conn = get_vision_runtime().encode(images, timings=timings)
        ref_dtype = next(policy.parameters()).dtype
        served = {"i": 0}

        def _npu_embed_image(image):
            i = served["i"]
            served["i"] += 1
            bsize = image.shape[0]
            emb = torch.from_numpy(np.ascontiguousarray(conn[i])).to(ref_dtype)
            return emb[None, ...].expand(bsize, -1, -1)

        vwe.embed_image = _npu_embed_image
        try:
            return orig_embed_prefix(*a, **kw)
        finally:
            vwe.embed_image = orig_embed_image

    if npu_vision:
        policy.model.embed_prefix = _wrapped_embed_prefix
    try:
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        policy.model.embed_prefix = orig_embed_prefix
        vwe.embed_image = orig_embed_image

    return chunk.detach().float().numpy()  # (1, chunk_size, action_dim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cpu", action="store_true", help="run the unmodified CPU model instead"
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()
    npu_vision = not args.cpu

    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy, args.prompt)
    if npu_vision:
        warmup_npu()

    timings: dict = {}
    t0 = time.perf_counter()
    chunk = run_hybrid_forward(
        batch,
        policy=policy,
        noise=fixed_noise(policy),
        npu_vision=npu_vision,
        timings=timings,
    )
    wall_ms = (time.perf_counter() - t0) * 1e3

    cfg = build_config(npu_vision)
    print(f"[smolvla] NPU stages   : {cfg['npu_stages']}")
    print(f"[smolvla] execution    : {cfg['execution_model']}")
    print(f"[smolvla] wall clock   : {wall_ms:.1f} ms")
    print(f"[smolvla] action chunk : {chunk.shape}  |x|max={np.abs(chunk).max():.4f}")
    for k, v in sorted(timings.items()):
        print(f"[smolvla]   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
