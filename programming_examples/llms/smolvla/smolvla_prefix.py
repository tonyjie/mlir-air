"""Dump SmolVLA backbone oracle fixtures from the real CPU model.
Run with the lerobot venv:  ~/Projects/smolvla_playground/.venv/bin/python
Writes smolvla_oracle.npz with: prefix_embed (241,960), per-layer hidden
(16 x (241,960)), final_norm_hidden (241,960), and the (50,6) action chunk.

IMPORTANT implementation note (found by inspecting
lerobot/policies/smolvla/smolvlm_with_expert.py): SmolVLAWithExpertModel.forward()
does NOT call the HF LlamaDecoderLayer modules as whole units (i.e. it never
does `layer(hidden_states, ...)`). Instead it manually re-implements one
attention+MLP step per layer by calling the layer's *submodules* directly
(`layer.input_layernorm`, `layer.self_attn.{q,k,v,o}_proj`, `layer.mlp`,
`layer.post_attention_layernorm`) and doing the residual adds itself in
plain Python. Consequently a forward_pre_hook/forward_hook registered on
`text_model.layers[i]` (the decoder layer module) NEVER FIRES for the real
select_action() code path -- the naive approach from the task spec (hooking
the whole decoder-layer module) silently captures nothing and
`np.stack([])` raises "need at least one array to stack".

Additionally, `text_model.norm` (the final RMSNorm) is never invoked at all
in this code path: the KV cache is built from raw (un-normed) per-layer
values, and there is no normalization step applied to the prefix stream
after the last decoder layer. So there is no naturally-occurring
"final_norm_hidden" tensor to capture from a hook either.

Fix applied here: hook the three submodules whose outputs are combined
(exactly as smolvlm_with_expert.py's forward() does) to reconstruct each
layer's post-residual hidden state without duplicating any numerics:
  hidden_states_input_i  <- forward_pre_hook on layer.input_layernorm
  o_proj_out_i            <- forward_hook on layer.self_attn.o_proj
  mlp_out_i               <- forward_hook on layer.mlp
  layer_output_i = hidden_states_input_i + o_proj_out_i + mlp_out_i
    (mirrors: out_emb = o_proj(att_out); out_emb += hidden_states;
     after_first_residual = out_emb.clone(); out_emb = mlp(post_attn_ln(out_emb));
     out_emb += after_first_residual)
`final_norm_hidden` is then computed explicitly as `text_model.norm(layer_output_15)`
via a direct module call (not a hook) -- this is the same RMSNorm module/weights
the real model owns, just invoked here for oracle purposes since production
inference never calls it on the prefix stream.
"""

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_ID = "lerobot/smolvla_base"
OUT = "smolvla_oracle.npz"


def build_batch(policy):
    cfg = policy.config
    b = {}
    for k, f in cfg.input_features.items():
        b[k] = torch.zeros((1, *tuple(f.shape)), dtype=torch.float32)
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        ["pick up the cube"],
        padding="max_length",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b


def main():
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).eval()
    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    n_layers = len(tm.layers)

    hidden_in = {}
    o_proj_out = {}
    mlp_out = {}
    hooks = []
    for i, layer in enumerate(tm.layers):
        hooks.append(
            layer.input_layernorm.register_forward_pre_hook(
                lambda m, args, idx=i: hidden_in.__setitem__(
                    idx, args[0].detach().clone()
                )
            )
        )
        hooks.append(
            layer.self_attn.o_proj.register_forward_hook(
                lambda m, i_, o, idx=i: o_proj_out.__setitem__(idx, o.detach().clone())
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda m, i_, o, idx=i: mlp_out.__setitem__(idx, o.detach().clone())
            )
        )

    batch = build_batch(policy)
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(batch)
    for h in hooks:
        h.remove()

    missing = [
        i
        for i in range(n_layers)
        if i not in hidden_in or i not in o_proj_out or i not in mlp_out
    ]
    if missing:
        raise RuntimeError(
            f"Failed to capture layer boundaries for layers {missing}; "
            f"hidden_in={sorted(hidden_in)} o_proj_out={sorted(o_proj_out)} mlp_out={sorted(mlp_out)}"
        )

    with torch.no_grad():
        # Downcast the residual input to the o_proj output dtype (bf16) before
        # summing. The real model accumulates the residual IN-PLACE on a bf16
        # tensor (`out_emb += hidden_states`), truncating to bf16 at that step.
        # For layer 0, hidden_in[0] is the raw fp32 embedding (no bf16 layer has
        # run yet), so an out-of-place Python `+` would trigger torch type
        # promotion and upcast the whole sum to fp32 -- keeping precision the real
        # inference never has (~5.36 max abs diff vs the true layer-0 output tapped
        # at layer[1].input_layernorm; matters for the layer-0 gate). Forcing bf16
        # here reproduces the in-place semantics; harmless for layers 1-15 (their
        # hidden_in is already a captured bf16 tensor).
        layer_outputs = [
            hidden_in[i].to(o_proj_out[i].dtype) + o_proj_out[i] + mlp_out[i]
            for i in range(n_layers)
        ]
        final_norm_hidden = tm.norm(layer_outputs[-1])

    prefix_embed = hidden_in[0].float().numpy()[0]
    layer_hidden = np.stack([t.float().numpy()[0] for t in layer_outputs])
    final_norm_hidden_np = final_norm_hidden.float().numpy()[0]

    np.savez(
        OUT,
        prefix_embed=prefix_embed,
        layer_hidden=layer_hidden,
        final_norm_hidden=final_norm_hidden_np,
        action=action.numpy(),
    )
    print(
        f"[oracle] wrote {OUT}: prefix{prefix_embed.shape} "
        f"layers{layer_hidden.shape} action{tuple(action.shape)}"
    )


if __name__ == "__main__":
    main()
