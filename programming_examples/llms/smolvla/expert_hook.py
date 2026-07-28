# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Splice the NPU action expert into a real lerobot SmolVLA policy (phase E5).

Minimal surface: only the 16-layer `vlm_with_expert.forward` call inside
`denoise_step` is replaced. lerobot's own `embed_suffix` and `action_out_proj`
still run, unmodified, on CPU -- so the head/tail is the real code path rather
than our numpy transcription of it, and the only thing under test is the part
that actually moved to the NPU.

Mirrors how `smolvla_inference.py` splices the NPU vision tower: wrap, capture,
substitute, restore in a `finally`.

Per-inference setup (prefix K/V re-projection, masks, RoPE LUTs) is keyed on the
identity of `past_key_values`, which lerobot rebuilds for every
`predict_action_chunk`. That is what makes the cross-attention K/V hoist safe:
it is recomputed exactly when the prefix changes and not once per denoise step.
"""

import numpy as np


def _to_np(t):
    import torch  # local import: this module is imported from the lerobot venv

    return t.detach().to(torch.float32).cpu().numpy()


def install_npu_expert(policy, runner, verbose=False):
    """Replace the expert's 16 layers with the NPU path. Returns a restore fn."""
    import torch

    m = policy.model
    cfg = policy.config
    orig_denoise_step = m.denoise_step
    state = {"pkv_id": None, "n_setup": 0, "n_steps": 0}

    def _setup(prefix_pad_masks, past_key_values, suffix_pad_masks, position_ids):
        """Everything that depends on the prefix but not on the denoise step."""
        n_layers = runner.cfg.n_layers
        pk = np.stack(
            [_to_np(past_key_values[i]["key_states"])[0] for i in range(n_layers)]
        )
        pv = np.stack(
            [_to_np(past_key_values[i]["value_states"])[0] for i in range(n_layers)]
        )
        prefix_len = pk.shape[1]

        # Rebuild exactly the masks denoise_step builds (modeling_smolvla.py).
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        suffix_len = suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(1, suffix_len, prefix_len)
        suffix_att = torch.ones(1, suffix_len, device=suffix_pad_masks.device)
        suffix_2d = make_att_2d_masks(suffix_pad_masks, suffix_att)
        full = torch.cat([prefix_2d, suffix_2d], dim=2)[0].cpu().numpy()

        pos = position_ids[0].cpu().numpy().astype(np.int64)
        runner.set_prefix(
            prefix_k=pk,
            prefix_v=pv,
            mask_self=full,
            mask_cross=full[:, :prefix_len],
            pos_self=pos,
            pos_cross=pos - pos.min(),
        )
        state["n_setup"] += 1
        if verbose:
            print(
                f"[expert] prefix set up (#{state['n_setup']}): "
                f"K/V {pk.shape}, mask_self {full.shape}, "
                f"pos {pos.min()}..{pos.max()}"
            )

    def npu_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        # lerobot's own head, untouched.
        suffix_embs, suffix_pad_masks, _ = m.embed_suffix(x_t, timestep)

        prefix_len = prefix_pad_masks.shape[1]
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        if state["pkv_id"] != id(past_key_values):
            _setup(prefix_pad_masks, past_key_values, suffix_pad_masks, position_ids)
            state["pkv_id"] = id(past_key_values)

        # The 16 layers + final norm, on NPU.
        out = runner.run_step(_to_np(suffix_embs)[0])
        state["n_steps"] += 1

        suffix_out = torch.from_numpy(np.ascontiguousarray(out))[None].to(
            dtype=torch.float32
        )
        # lerobot's own tail, untouched.
        return m.action_out_proj(suffix_out)

    m.denoise_step = npu_denoise_step

    def restore():
        m.denoise_step = orig_denoise_step
        return dict(state)

    return restore
