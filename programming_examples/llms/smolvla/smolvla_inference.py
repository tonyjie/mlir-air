# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA end-to-end hybrid inference (CPU prefix -> NPU backbone -> CPU expert).

Runs in the LEROBOT venv (`~/Projects/smolvla_playground/.venv/bin/python`),
which has torch + lerobot but NOT air/pyxrt. The NPU work runs in the WORKTREE
python (`.../sandbox/bin/python3`, has air/pyxrt but not lerobot) via subprocess
bridges exchanging tensors through .npz files.

Execution model = BRIDGED (two processes). Chosen because the two Python envs
are genuinely disjoint (verified: lerobot venv has no `air`, worktree python has
no `torch`/`lerobot`), so a single-process design is impossible. Each bridge is
a single synchronous subprocess call inside a wrapped lerobot method.

Pipeline (run_hybrid_forward):
  1. Vision (3 cameras -> 3 x (64,960) image embeddings):
       npu_vision=False (default): CPU lerobot SigLIP + connector, untouched.
       npu_vision=True : ONE run_npu_vision.py subprocess encodes ALL 3 images
         on NPU (12-layer ViT + connector GEMM); `vlm_with_expert.embed_image`
         is swapped to serve those results. Everything after it — the sqrt(960)
         scale, language/state tokens, prefix assembly — stays lerobot's.
  2. CPU (lerobot): prefix assembly -> (241,960) prefix embed via the real
     embed_prefix, NOT reimplemented.
  3. NPU (worktree python subprocess): run_backbone_prefill on the prefix embed
     -> per-layer post-RoPE K + raw V (5 kv-heads), via run_npu_backbone.py.
  4. CPU (lerobot): the real sample_actions() builds a correctly-structured
     past_key_values (fill_kv_cache=True), which we OVERWRITE in place with the
     NPU K/V, then the unchanged 10-step denoise loop reads it -> (1,50,6) chunk.

The NPU-lock discipline: the OUTER caller (test_e2e / make verify) holds
`flock /tmp/mlir-air-npu.lock`. The subprocesses here do NOT re-lock that path
(would self-deadlock); their KernelCache uses a distinct inner filelock.

Timing attribution: pass a dict as `timings=` to collect, per bridge, the wall
time the driver spent (spawn + npz I/O + subprocess) AND the subprocess-internal
phase breakdown (weight load / ELF load / actual NPU compute), so bench_e2e.py
can report both "as measured" and "compute-only" numbers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
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


def build_config(npu_vision: bool = False):
    """Minimal config dict for the verify adapter / reporting."""
    return {
        "model": DEFAULT_MODEL,
        "prompt": DEFAULT_PROMPT,
        "execution_model": "bridged (lerobot-venv driver + worktree-python NPU)",
        "seq_len": 256,
        "n_layers": 16,
        "npu_stages": "vision+backbone" if npu_vision else "backbone",
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


def _record_bridge_timings(timings, stage, wall_s, out):
    """Fold one bridge invocation's timing into the caller's `timings` dict.

    wall_s      : what the DRIVER paid (npz write + spawn + subprocess + npz read).
    out[t_*_ms] : what the SUBPROCESS spent per phase (see bridge_common.Timings).
    The difference (wall - subprocess total) is pure process-spawn + interpreter
    startup + import cost, which a single-process deployment would not pay."""
    if timings is None:
        return
    rec = {"wall_ms": wall_s * 1e3}
    for k in out.files:
        if k.startswith("t_") and k.endswith("_ms"):
            rec[k] = np.asarray(out[k]).tolist()
    rec["spawn_ms"] = rec["wall_ms"] - float(rec.get("t_bridge_total_ms", 0.0))
    timings[stage] = rec


def _run_npu_backbone(
    prefix_embed, pad_mask, position_ids, workdir, attn_mode="gemm", timings=None
):
    """Bridge: write the prefix fixture, invoke run_npu_backbone.py in the
    worktree python, read back per-layer K/V. Synchronous.

    attn_mode: "gemm" (default, approach-B, the production NPU attention
    path) or "flash" (EXPERIMENT: registry FlashAttention, non-causal, no
    mask -- see smolvla_backbone_prefill.py's attn_mode="flash" docstring)."""
    assert attn_mode in ("gemm", "flash"), attn_mode
    in_path = str(Path(workdir) / "npu_in.npz")
    out_path = str(Path(workdir) / "npu_out.npz")
    t0 = time.perf_counter()
    np.savez(
        in_path,
        prefix_embed=np.asarray(prefix_embed, np.float32),
        pad_mask=np.asarray(pad_mask, bool),
        position_ids=np.asarray(position_ids, np.int64),
    )
    cmd = [
        WORKTREE_PYTHON,
        str(_HERE / "run_npu_backbone.py"),
        in_path,
        out_path,
        attn_mode,
    ]
    print(f"[hybrid] NPU backbone subprocess: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    out = np.load(out_path)
    _record_bridge_timings(timings, "backbone", time.perf_counter() - t0, out)
    return out["k"], out["v"]  # (L, oracle_len, 5, 64) f32


def _run_npu_vision(images, workdir, timings=None):
    """Bridge: encode ALL cameras on NPU in ONE subprocess call.

    images: list/sequence of N torch tensors, each (B=1, 3, 512, 512), exactly
    the tensors lerobot's `prepare_images` produced (already resized-with-pad
    and scaled to [-1, 1]) and would have fed to `embed_image` one at a time.

    Returns a numpy array (N, 64, 960) of RAW connector outputs — the same
    quantity `embed_image` returns on CPU, BEFORE embed_prefix's sqrt(960)
    scale. One invocation for all N so the process spawn, the safetensors
    weight load and the ELF/XRT setup are paid once instead of N times."""
    stacked = np.stack(
        [np.asarray(img.detach().float().numpy()[0], np.float32) for img in images]
    )  # (N, 3, 512, 512)
    in_path = str(Path(workdir) / "npu_vision_in.npz")
    out_path = str(Path(workdir) / "npu_vision_out.npz")
    t0 = time.perf_counter()
    np.savez(in_path, images=stacked)
    cmd = [WORKTREE_PYTHON, str(_HERE / "run_npu_vision.py"), in_path, out_path]
    print(
        f"[hybrid] NPU vision subprocess ({stacked.shape[0]} images): "
        f"{' '.join(cmd)}",
        flush=True,
    )
    subprocess.run(cmd, check=True)
    out = np.load(out_path)
    _record_bridge_timings(timings, "vision", time.perf_counter() - t0, out)
    conn = np.asarray(out["connector"], np.float32)
    assert conn.shape[0] == stacked.shape[0], (conn.shape, stacked.shape)
    return conn


def run_hybrid_forward(
    batch,
    policy=None,
    noise=None,
    workdir=None,
    attn_mode="gemm",
    npu_vision=False,
    timings=None,
):
    """Run the full hybrid pipeline once; return the (1, chunk, action_dim)
    action chunk (unpadded to the real action dim).

    attn_mode: forwarded to _run_npu_backbone -- "gemm" (default, approach-B,
        production) or "flash" (EXPERIMENT: non-causal FlashAttention, no mask).
    npu_vision: if True, the 3 camera images are encoded by the NPU SigLIP ViT +
        connector (one run_npu_vision.py subprocess for all of them) instead of
        lerobot's CPU vision tower. Default False keeps the CPU vision path as
        the reference. The NPU backbone runs either way.
    timings: optional dict; filled with per-bridge wall/phase timings (see
        _record_bridge_timings)."""
    if policy is None:
        policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    tmpdir_ctx = None
    if workdir is None:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        workdir = tmpdir_ctx.name

    vwe = policy.model.vlm_with_expert
    n_layers = len(vwe.get_vlm_model().text_model.layers)

    # -- Capture the assembled prefix embed + pad masks; optionally serve the
    #    per-camera image embeddings from NPU. --
    orig_embed_prefix = policy.model.embed_prefix
    orig_embed_image = vwe.embed_image
    captured = {}

    def _wrapped_embed_prefix(*a, **kw):
        if npu_vision:
            # `images` is embed_prefix's first positional arg (a list of N
            # camera tensors). Encode them ALL in one bridge call up front, then
            # let the untouched embed_prefix pull them one at a time through the
            # swapped embed_image -- so the sqrt(960) scale, the pad/att masks
            # and the prefix assembly all stay lerobot's own code.
            images = kw["images"] if "images" in kw else a[0]
            conn = _run_npu_vision(images, workdir, timings=timings)
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
                embs, pad_masks, att_masks = orig_embed_prefix(*a, **kw)
            finally:
                vwe.embed_image = orig_embed_image
            assert served["i"] == len(images), (served["i"], len(images))
        else:
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
                captured["prefix_embed"],
                captured["pad_masks"],
                position_ids,
                workdir,
                attn_mode=attn_mode,
                timings=timings,
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
        vwe.embed_image = orig_embed_image
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
    npu_vision = "--npu-vision" in sys.argv
    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()
    batch = build_oracle_batch(policy)
    timings = {}
    chunk = run_hybrid_forward(
        batch,
        policy=policy,
        noise=_fixed_noise(policy),
        npu_vision=npu_vision,
        timings=timings,
    )
    print(f"[hybrid] npu stages: {build_config(npu_vision)['npu_stages']}")
    print(f"[hybrid] action chunk shape {chunk.shape}")
    print(f"[hybrid] chunk[0,0] = {np.asarray(chunk[0, 0], np.float32)}")
    for stage, rec in timings.items():
        print(f"[hybrid] bridge {stage}: {rec}")


if __name__ == "__main__":
    main()
