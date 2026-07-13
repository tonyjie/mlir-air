# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end hybrid inference (CPU prefix -> NPU backbone -> CPU expert).

Runs in the LEROBOT venv (`~/Projects/smolvla_playground/.venv/bin/python`),
which has torch + lerobot but NOT air/pyxrt. The NPU backbone prefill runs in
the WORKTREE python (`.../sandbox/bin/python3`, has air/pyxrt but not lerobot)
via a subprocess bridge exchanging tensors through .npz files.

Execution model = BRIDGED (two processes). Chosen because the two Python envs
are genuinely disjoint (verified: lerobot venv has no `air`, worktree python has
no `torch`/`lerobot`), so a single-process design is impossible. The bridge is a
single synchronous subprocess call inside the wrapped vlm_with_expert.forward.

Pipeline (run_hybrid_forward):
  1. CPU (lerobot): vision -> connector -> prefix assembly -> (241,960) prefix
     embed. Reuses the real model's embed_prefix (SigLIP etc.), NOT reimplemented.
  2. NPU (worktree python subprocess): run_backbone_prefill on the prefix embed
     -> per-layer post-RoPE K + raw V (5 kv-heads), via run_npu_backbone.py.
  3. CPU (lerobot): the real sample_actions() builds a correctly-structured
     past_key_values (fill_kv_cache=True), which we OVERWRITE in place with the
     NPU K/V, then the unchanged 10-step denoise loop reads it -> (1,50,6) chunk.

The NPU-lock discipline: the OUTER caller (test_e2e / make verify) holds
`flock /tmp/mlir-air-npu.lock`. The subprocess here does NOT re-lock that path
(would self-deadlock); run_npu_backbone.py's own KernelCache uses a distinct
inner filelock.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

_HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = "lerobot/smolvla_base"
# Worktree python (has air/pyxrt). Overridable for portability.
WORKTREE_PYTHON = os.environ.get(
    "SMOLVLA_NPU_PYTHON", "/home/jiajli/apps/mlir-air/sandbox/bin/python3"
)
DEFAULT_PROMPT = "pick up the cube"


def normalized_mse(chunk, ref) -> float:
    """MSE relative to the baseline action's own power: mean((chunk-ref)**2) /
    mean(ref**2). Magnitude-invariant, so the gate does not depend on the
    absolute scale of a particular prompt's action chunk (a raw MSE_MAX would
    silently drift PASS/FAIL as action magnitude changes). Single source of
    truth shared by test_e2e.py and verify_adapter.py."""
    chunk = np.asarray(chunk, np.float32)
    ref = np.asarray(ref, np.float32)
    power = float(np.mean(ref**2))
    return float(np.mean((chunk - ref) ** 2) / max(power, 1e-12))


def build_config():
    """Minimal config dict for the verify adapter / reporting."""
    return {
        "model": DEFAULT_MODEL,
        "prompt": DEFAULT_PROMPT,
        "execution_model": "bridged (lerobot-venv driver + worktree-python NPU)",
        "seq_len": 256,
        "n_layers": 16,
    }


def build_oracle_batch(policy, prompt: str = DEFAULT_PROMPT):
    """Build the same synthetic batch the oracle dumper uses (zeros images +
    state, tokenized prompt)."""
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


def _run_npu_backbone(prefix_embed, pad_mask, position_ids, workdir):
    """Bridge: write the prefix fixture, invoke run_npu_backbone.py in the
    worktree python, read back per-layer K/V. Synchronous."""
    in_path = str(Path(workdir) / "npu_in.npz")
    out_path = str(Path(workdir) / "npu_out.npz")
    np.savez(
        in_path,
        prefix_embed=np.asarray(prefix_embed, np.float32),
        pad_mask=np.asarray(pad_mask, bool),
        position_ids=np.asarray(position_ids, np.int64),
    )
    cmd = [WORKTREE_PYTHON, str(_HERE / "run_npu_backbone.py"), in_path, out_path]
    print(f"[hybrid] NPU backbone subprocess: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    out = np.load(out_path)
    return out["k"], out["v"]  # (L, oracle_len, 5, 64) f32


def run_hybrid_forward(batch, policy=None, noise=None, workdir=None):
    """Run the full hybrid pipeline once; return the (1, chunk, action_dim)
    action chunk (unpadded to the real action dim)."""
    if policy is None:
        policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    tmpdir_ctx = None
    if workdir is None:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        workdir = tmpdir_ctx.name

    vwe = policy.model.vlm_with_expert
    n_layers = len(vwe.get_vlm_model().text_model.layers)

    # -- Capture the assembled prefix embed + pad masks (CPU vision/connector). --
    orig_embed_prefix = policy.model.embed_prefix
    captured = {}

    def _wrapped_embed_prefix(*a, **kw):
        embs, pad_masks, att_masks = orig_embed_prefix(*a, **kw)
        captured["prefix_embed"] = embs.detach().float().numpy()[0]
        captured["pad_masks"] = pad_masks.detach().bool().numpy()[0]
        return embs, pad_masks, att_masks

    policy.model.embed_prefix = _wrapped_embed_prefix

    # -- Overwrite the CPU KV cache with NPU K/V on the fill call. --
    orig_vwe_forward = vwe.forward

    def _wrapped_vwe_forward(*a, **kw):
        out = orig_vwe_forward(*a, **kw)
        if kw.get("fill_kv_cache", False):
            pkv = out[1]
            position_ids = kw["position_ids"].detach().numpy()[0].astype(np.int64)
            npu_k, npu_v = _run_npu_backbone(
                captured["prefix_embed"], captured["pad_masks"], position_ids, workdir
            )
            # Reassign whole contiguous tensors (recipe: don't mutate views).
            # Match each cached tensor's dtype/shape exactly: (1, L, 5, 64) bf16.
            for i in range(n_layers):
                ref_k = pkv[i]["key_states"]
                ref_v = pkv[i]["value_states"]
                k_t = torch.from_numpy(np.ascontiguousarray(npu_k[i])).to(
                    dtype=ref_k.dtype
                )[None, ...]
                v_t = torch.from_numpy(np.ascontiguousarray(npu_v[i])).to(
                    dtype=ref_v.dtype
                )[None, ...]
                assert k_t.shape == ref_k.shape, (k_t.shape, ref_k.shape)
                assert v_t.shape == ref_v.shape, (v_t.shape, ref_v.shape)
                pkv[i]["key_states"] = k_t
                pkv[i]["value_states"] = v_t
        return out

    vwe.forward = _wrapped_vwe_forward

    try:
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        policy.model.embed_prefix = orig_embed_prefix
        vwe.forward = orig_vwe_forward
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    return chunk.detach().float().numpy()  # (1, chunk_size, action_dim)


def _fixed_noise(policy):
    """Deterministic zero noise matching the oracle's action_chunk baseline."""
    return torch.zeros(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        dtype=torch.float32,
    )


def main():
    """Standalone: run one hybrid forward and print the action chunk summary."""
    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    chunk = run_hybrid_forward(batch, policy=policy, noise=_fixed_noise(policy))
    print(f"[hybrid] action chunk shape {chunk.shape}")
    print(f"[hybrid] chunk[0,0] = {np.asarray(chunk[0, 0], np.float32)}")


if __name__ == "__main__":
    main()
