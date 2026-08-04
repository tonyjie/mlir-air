# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the unmodified CPU model and save what `make verify` compares against.

`make oracle` -> smolvla_oracle.npz, holding the (1, 50, 6) action chunk: the
tensor the robot would actually execute. Needs torch + lerobot (see
requirements.txt); no NPU.

The baseline is upstream lerobot itself, not a reimplementation in this repo,
so the gate measures "what does swapping in the NPU vision encoder change"
rather than "do two of my own implementations agree".

The noise is pinned to zero because the action expert is a flow-matching
denoiser seeded from noise; without that the gate would measure sampling
variance. `build_oracle_batch` is imported rather than copied so the oracle and
the gate can never drift onto different inputs.

Porting another stage to the NPU
--------------------------------
This file used to dump eight more arrays -- per-layer backbone hidden states,
the KV cache, prefix pad masks, position ids -- for the backbone and action
expert ports, which measured slower than the CPU and are not part of this
example. That code, and the NPU implementations that consumed it, are on the
`smolvla` branch. See docs/code_walkthrough.md for the recipe and the two traps
worth knowing before you re-derive it.
"""

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smolvla_inference import DEFAULT_MODEL, build_oracle_batch, fixed_noise

OUT = "smolvla_oracle.npz"


def main():
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    policy.reset()
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(
            build_oracle_batch(policy), noise=fixed_noise(policy)
        )

    action_chunk = action_chunk.detach().float().numpy()
    np.savez(OUT, action_chunk=action_chunk)
    print(f"[oracle] wrote {OUT}: action_chunk{action_chunk.shape}")


if __name__ == "__main__":
    main()
