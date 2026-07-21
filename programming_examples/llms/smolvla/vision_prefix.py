"""Dump SmolVLA vision-encoder (SigLIP ViT) oracle fixtures from the real CPU model.
Run with the lerobot venv:  ~/Projects/smolvla_playground/.venv/bin/python
Writes vision_oracle.npz with: pixel_values (3,512,512), patch_embed (1024,768),
per-layer hidden (12 x (1024,768)), post_ln (1024,768), connector (64,960),
connector_scaled (64,960).

Unlike the backbone (smolvla_prefix.py), the vision path has NO hook trap:
SmolVLMEncoder.forward loops `encoder_layer(hidden_states, ...)` calling each
SmolVLMEncoderLayer as a whole module, so a plain forward_hook on
`vision_model.encoder.layers[i]` fires and its output IS the post-residual
hidden state of that layer. `embeddings`, `post_layernorm`, `connector` are
likewise whole-module hooks.

The one thing with no module hook point is the pixel_values tensor that actually
enters the vision model (after lerobot's resize_with_pad to 512x512). We grab it
with a forward_pre_hook on the vision_model module itself.

The connector output the *backbone* consumes is scaled by sqrt(hidden_dim=960)
in embed_prefix AFTER embed_image; the connector-module hook captures it BEFORE
that scale. We save both (connector raw, connector_scaled) to avoid confusion.
"""

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_ID = "lerobot/smolvla_base"
OUT = "vision_oracle.npz"
HIDDEN_SCALE = 960.0  # embed_prefix multiplies img emb by sqrt(960)


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
    vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
    connector = policy.model.vlm_with_expert.get_vlm_model().connector
    n_layers = len(vm.encoder.layers)

    captured = {}
    hooks = []

    # pixel_values entering the vision model (forward_pre_hook: args[0] or kwarg)
    def _vm_pre(m, args, kwargs):
        pv = None
        if args:
            pv = args[0]
        elif "pixel_values" in kwargs:
            pv = kwargs["pixel_values"]
        if pv is not None and "pixel_values" not in captured:
            captured["pixel_values"] = pv.detach().float().clone()

    hooks.append(vm.register_forward_pre_hook(_vm_pre, with_kwargs=True))

    # embeddings output (patch conv + pos emb)
    hooks.append(
        vm.embeddings.register_forward_hook(
            lambda m, i, o: captured.__setitem__(
                "patch_embed", o.detach().float().clone()
            )
        )
    )
    # per-layer hidden states (encoder layer output may be a tuple)
    layer_out = {}
    for i, layer in enumerate(vm.encoder.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda m, inp, o, idx=i: layer_out.__setitem__(
                    idx, (o[0] if isinstance(o, tuple) else o).detach().float().clone()
                )
            )
        )
    # post_layernorm output (final encoder output)
    hooks.append(
        vm.post_layernorm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("post_ln", o.detach().float().clone())
        )
    )
    # connector output (pre sqrt(960) scale)
    hooks.append(
        connector.register_forward_hook(
            lambda m, i, o: captured.__setitem__(
                "connector", o.detach().float().clone()
            )
        )
    )

    batch = build_batch(policy)
    policy.reset()
    with torch.no_grad():
        policy.select_action(batch)
    for h in hooks:
        h.remove()

    missing_layers = [i for i in range(n_layers) if i not in layer_out]
    if missing_layers:
        raise RuntimeError(f"Failed to capture vision layers {missing_layers}")
    for key in ("pixel_values", "patch_embed", "post_ln", "connector"):
        if key not in captured:
            raise RuntimeError(f"Failed to capture {key}")

    # Drop batch dim. pixel_values is (B, num_images, 3, H, W) or (B, 3, H, W);
    # the vision model flattens images into the batch axis, so take the first
    # image of the first sample.
    pv = captured["pixel_values"].numpy()
    while pv.ndim > 3:
        pv = pv[0]
    pixel_values = pv  # (3, 512, 512)

    patch_embed = captured["patch_embed"].numpy()[0]  # (1024, 768)
    layer_hidden = np.stack(
        [layer_out[i].numpy()[0] for i in range(n_layers)]
    )  # (12,1024,768)
    post_ln = captured["post_ln"].numpy()[0]  # (1024, 768)
    connector = captured["connector"].numpy()[0]  # (64, 960)
    connector_scaled = connector * np.sqrt(HIDDEN_SCALE).astype(np.float32)

    np.savez(
        OUT,
        pixel_values=pixel_values,
        patch_embed=patch_embed,
        layer_hidden=layer_hidden,
        post_ln=post_ln,
        connector=connector,
        connector_scaled=connector_scaled,
    )
    print(
        f"[vision-oracle] wrote {OUT}: pixel{pixel_values.shape} "
        f"patch{patch_embed.shape} layers{layer_hidden.shape} "
        f"post_ln{post_ln.shape} connector{connector.shape}"
    )


if __name__ == "__main__":
    main()
