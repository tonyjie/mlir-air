"""Dump SmolVLA action-expert oracle fixtures from the real CPU model.
Run with the lerobot venv:  ~/Projects/smolvla_playground/.venv/bin/python
Writes expert_oracle.npz: per denoise-step x per-layer hidden in/out, the
attention intermediates (q after RoPE, k, v, attention output before o_proj),
the masks/positions, the final-norm output, v_t, and the (1,50,6) action chunk.

The expert (verified against lerobot/smolvla_base, not recalled):
hidden 720, intermediate 2048, 16 layers, GQA 15 q / 5 kv, head_dim 64,
rms_norm_eps 1e-5, no attention bias, RoPE base 10000 (apply_rope's default --
smolvlm_with_expert.py never passes max_wavelength). 50 action tokens,
config.num_steps=10 denoise iterations, run strictly sequentially.
self_attn_every_n_layers=2 -> EVEN layer indices self-attend, ODD cross-attend.

## What is captured, and why it cannot be captured the obvious way

Same hook trap as the backbone (see smolvla_prefix.py): SmolVLMWithExpertModel
.forward() never calls the decoder layer as a module, it drives the submodules
by hand. So we hook `layer.input_layernorm` (pre), `layer.self_attn.o_proj`
and `layer.mlp`, and rebuild
    layer_out = hidden_in.to(bf16) + o_proj_out + mlp_out
mirroring `out_emb += hidden_states; ...; out_emb += after_first_residual`.
The `.to(bf16)` reproduces the in-place bf16 truncation of the real residual
add (matters for layer 0, whose input is the fp32 embed_suffix output).

Attention intermediates have no module hook point at all, so we monkeypatch
`vwe.eager_attention_forward` (q/k/v/att_output) and the two layer drivers
`forward_attn_layer` / `forward_cross_attn_layer` (to know which layer_idx the
interface call belongs to). During the prefix fill pass the same interface runs
for the *backbone*; captures are gated on a step counter that is -1 until the
first embed_suffix call.

## EVEN (self-attn) layers
K/V = the 241-token backbone prefix cache (past_key_values[layer_idx], K already
RoPE'd at prefix positions) concatenated with the 50 new action-token K/V ->
291. Q/K of the new tokens get RoPE at positions 197..246 (prefix_offsets +
cumsum(suffix_pad)-1, NOT re-based to 0).

## ODD (cross-attn) layers
q is over the 50 action tokens with positions RE-BASED to 0..49
(`expert_position_id - min(expert_position_id)`), but k_proj/v_proj are fresh
nn.Linear(320,320) run over the **241-token prefix cache** and recomputed every
denoise step. K gets NO RoPE on this path. Those k_proj/v_proj weights are
**fp32** while every other expert weight is bf16 (confirmed in the safetensors
index). Because their input is the fixed prefix cache, the result is
step-invariant -- asserted here, and stored once without a step axis.

## Mask -- CORRECTION to the "prefix columns are all-True" assumption
The prefix columns of the denoise mask are `prefix_pad_masks` broadcast over
rows, and prefix_pad_masks is NOT all-True: the language block is tokenized
with padding="max_length" (48 slots) so for the oracle prompt only 197 of 241
prefix tokens are real. Columns 196..239 are False on BOTH the self and the
cross path. Only the suffix block (columns 241..290, self layers only) is
causal among the 50 action tokens. Asserted below so a prompt change breaks
loudly rather than silently shifting the mask.

## Size
All 10 steps are kept for every tensor. The per-step axis is affordable only
because the two big K/V tensors are stored deduplicated: the 241-token prefix
cache once as (16,241,5,64), the odd-layer projected K/V once as (8,241,5,64)
(step-invariant, asserted), and the self-attn K/V per step only for the 50 NEW
tokens. Materializing the full 291-token K/V per step per layer instead would
cost ~600 MB; this layout is ~150 MB.
"""

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

MODEL_ID = "lerobot/smolvla_base"
OUT = "expert_oracle.npz"

N_LAYERS = 16
N_STEPS = 10
SEQ = 50  # action tokens (chunk_size)
PREFIX_LEN = 241  # backbone prefix cache length (vision + language + state)
N_KV_HEADS = 5
N_Q_HEADS = 15
HEAD_DIM = 64
Q_DIM = N_Q_HEADS * HEAD_DIM  # 960
SELF_LAYERS = list(range(0, N_LAYERS, 2))
CROSS_LAYERS = list(range(1, N_LAYERS, 2))


def build_batch(policy):
    """Same fixed batch as smolvla_prefix.py / vision_prefix.py: zero inputs for
    every declared feature plus the fixed "pick up the cube" prompt."""
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


def _f32(t):
    return t.detach().float().numpy()


class _Capture:
    """Collects everything the denoise loop produces, keyed by (step, layer)."""

    def __init__(self):
        self.step = -1  # -1 = prefix fill pass, captures disabled
        self.layer = None
        self.hidden_in = {}
        self.o_proj_out = {}
        self.mlp_out = {}
        self.q_rope = {}
        self.k_full = {}
        self.v_full = {}
        self.att_out = {}
        self.mask = {}
        self.final_norm = {}
        self.suffix_embs = {}
        self.x_t = {}
        self.timestep = {}
        self.v_t = {}
        self.position_ids = {}
        self.att_mask_full = {}
        self.prefix_kv = None

    @property
    def active(self):
        return self.step >= 0


def _install(policy, cap):
    """Hook/patch everything; returns a restore() callable."""
    vwe = policy.model.vlm_with_expert
    expert = vwe.lm_expert
    hooks = []
    undo = []

    for i, layer in enumerate(expert.layers):
        hooks.append(
            layer.input_layernorm.register_forward_pre_hook(
                lambda m, args, idx=i: cap.active
                and cap.hidden_in.__setitem__((cap.step, idx), args[0].detach().clone())
            )
        )
        hooks.append(
            layer.self_attn.o_proj.register_forward_hook(
                lambda m, i_, o, idx=i: cap.active
                and cap.o_proj_out.__setitem__((cap.step, idx), o.detach().clone())
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda m, i_, o, idx=i: cap.active
                and cap.mlp_out.__setitem__((cap.step, idx), o.detach().clone())
            )
        )
    hooks.append(
        expert.norm.register_forward_hook(
            lambda m, i_, o: cap.active
            and cap.final_norm.__setitem__(cap.step, o.detach().clone())
        )
    )

    # embed_suffix: marks the start of a denoise step and yields x_t / timestep.
    orig_embed_suffix = policy.model.embed_suffix

    def _embed_suffix(noisy_actions, timestep, *a, **kw):
        cap.step += 1
        out = orig_embed_suffix(noisy_actions, timestep, *a, **kw)
        cap.x_t[cap.step] = noisy_actions.detach().clone()
        cap.timestep[cap.step] = float(timestep.reshape(-1)[0])
        cap.suffix_embs[cap.step] = out[0].detach().clone()
        return out

    policy.model.embed_suffix = _embed_suffix
    undo.append(lambda: setattr(policy.model, "embed_suffix", orig_embed_suffix))

    # denoise_step: v_t per step.
    orig_denoise = policy.model.denoise_step

    def _denoise(*a, **kw):
        out = orig_denoise(*a, **kw)
        cap.v_t[cap.step] = out.detach().clone()
        return out

    policy.model.denoise_step = _denoise
    undo.append(lambda: setattr(policy.model, "denoise_step", orig_denoise))

    # vwe.forward: prefix KV cache (fill call) and the denoise mask/positions.
    orig_forward = vwe.forward

    def _forward(*a, **kw):
        out = orig_forward(*a, **kw)
        if kw.get("fill_kv_cache", False):
            cap.prefix_kv = out[1]
        elif cap.active:
            cap.position_ids[cap.step] = kw["position_ids"].detach().clone()
            cap.att_mask_full[cap.step] = kw["attention_mask"].detach().clone()
        return out

    vwe.forward = _forward
    undo.append(lambda: setattr(vwe, "forward", orig_forward))

    # The two layer drivers only tell us which layer the next interface call
    # belongs to; the interface itself does the capturing.
    for name in ("forward_attn_layer", "forward_cross_attn_layer"):
        orig = getattr(vwe, name)

        def _wrap(*a, _orig=orig, **kw):
            cap.layer = kw["layer_idx"] if "layer_idx" in kw else a[2]
            return _orig(*a, **kw)

        setattr(vwe, name, _wrap)
        undo.append(lambda n=name, o=orig: setattr(vwe, n, o))

    orig_attn = vwe.eager_attention_forward

    def _attn(attention_mask, batch_size, head_dim, q, k, v):
        out = orig_attn(attention_mask, batch_size, head_dim, q, k, v)
        if cap.active:
            key = (cap.step, cap.layer)
            cap.q_rope[key] = q.detach().clone()
            cap.k_full[key] = k.detach().clone()
            cap.v_full[key] = v.detach().clone()
            cap.att_out[key] = out.detach().clone()
            cap.mask[key] = attention_mask.detach().clone()
        return out

    vwe.eager_attention_forward = _attn
    undo.append(lambda: setattr(vwe, "eager_attention_forward", orig_attn))

    def restore():
        for h in hooks:
            h.remove()
        for f in undo:
            f()

    return restore


def main():
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).eval()
    cfg = policy.config
    if (cfg.chunk_size, cfg.num_steps) != (SEQ, N_STEPS):
        raise RuntimeError(
            f"chunk_size/num_steps changed: {cfg.chunk_size}/{cfg.num_steps}"
        )
    if cfg.self_attn_every_n_layers != 2:
        raise RuntimeError(
            f"self_attn_every_n_layers={cfg.self_attn_every_n_layers}, expected 2"
        )

    cap = _Capture()
    restore = _install(policy, cap)
    try:
        batch = build_batch(policy)
        policy.reset()
        # Fixed ZERO noise, matching smolvla_inference._fixed_noise and the
        # backbone oracle's action_chunk baseline -- makes the dump deterministic.
        noise = torch.zeros(
            (1, cfg.chunk_size, cfg.max_action_dim), dtype=torch.float32
        )
        with torch.no_grad():
            action_chunk = policy.predict_action_chunk(batch, noise=noise)
    finally:
        restore()

    if cap.step != N_STEPS - 1:
        raise RuntimeError(f"saw {cap.step + 1} denoise steps, expected {N_STEPS}")
    if cap.prefix_kv is None:
        raise RuntimeError("failed to capture the prefix KV cache (fill pass)")

    # ---- structural assertions: break loudly if the model changes -----------
    prefix_k = np.stack(
        [_f32(cap.prefix_kv[i]["key_states"])[0] for i in range(N_LAYERS)]
    )  # (16,241,5,64)
    prefix_v = np.stack(
        [_f32(cap.prefix_kv[i]["value_states"])[0] for i in range(N_LAYERS)]
    )
    if prefix_k.shape != (N_LAYERS, PREFIX_LEN, N_KV_HEADS, HEAD_DIM):
        raise RuntimeError(f"prefix cache shape {prefix_k.shape}")

    for s in range(N_STEPS):
        for L in range(N_LAYERS):
            k = cap.k_full[(s, L)]
            m = cap.mask[(s, L)]
            want_kv = PREFIX_LEN + SEQ if L % 2 == 0 else PREFIX_LEN
            if tuple(k.shape) != (1, want_kv, N_KV_HEADS, HEAD_DIM):
                raise RuntimeError(f"step {s} layer {L}: k shape {tuple(k.shape)}")
            if tuple(m.shape) != (1, SEQ, want_kv):
                raise RuntimeError(f"step {s} layer {L}: mask shape {tuple(m.shape)}")
            if L % 2 == 0:
                # self layer: the first 241 K/V columns ARE the backbone cache
                np.testing.assert_array_equal(
                    _f32(k)[0, :PREFIX_LEN], prefix_k[L], f"step {s} L{L} k prefix"
                )
                np.testing.assert_array_equal(
                    _f32(cap.v_full[(s, L)])[0, :PREFIX_LEN],
                    prefix_v[L],
                    f"step {s} L{L} v prefix",
                )
            else:
                # cross layer: K/V depend only on the fixed prefix cache
                np.testing.assert_array_equal(
                    _f32(k), _f32(cap.k_full[(0, L)]), f"step {s} L{L} k not invariant"
                )
                np.testing.assert_array_equal(
                    _f32(cap.v_full[(s, L)]),
                    _f32(cap.v_full[(0, L)]),
                    f"step {s} L{L} v not invariant",
                )

    mask_self = cap.mask[(0, 0)].numpy()[0]  # (50,291) bool
    mask_cross = cap.mask[(0, 1)].numpy()[0]  # (50,241) bool
    # Every row shares the same prefix-column pattern == prefix_pad_masks, and
    # the cross mask is exactly that pattern (no suffix block).
    prefix_pad_masks = mask_self[0, :PREFIX_LEN].copy()
    if not np.all(mask_self[:, :PREFIX_LEN] == prefix_pad_masks[None, :]):
        raise RuntimeError("self-layer prefix columns are not row-invariant")
    if not np.all(mask_cross == prefix_pad_masks[None, :]):
        raise RuntimeError("cross-layer mask != prefix pad mask broadcast")
    if not np.array_equal(
        mask_self[:, PREFIX_LEN:], np.tril(np.ones((SEQ, SEQ), bool))
    ):
        raise RuntimeError("suffix block of the self mask is not lower-triangular")
    for s in range(N_STEPS):
        for L in range(N_LAYERS):
            ref = mask_self if L % 2 == 0 else mask_cross
            if not np.array_equal(cap.mask[(s, L)].numpy()[0], ref):
                raise RuntimeError(f"step {s} layer {L}: mask differs")

    pos_self = cap.position_ids[0].numpy()[0].astype(np.int64)  # 197..246
    pos_cross = pos_self - pos_self.min()  # 0..49, re-based
    if not np.array_equal(pos_self, pos_self[0] + np.arange(SEQ)):
        raise RuntimeError(f"self positions are not contiguous: {pos_self}")
    if int(pos_self[0]) != int(prefix_pad_masks.sum()):
        raise RuntimeError(
            f"self position offset {pos_self[0]} != real prefix tokens "
            f"{int(prefix_pad_masks.sum())}"
        )
    for s in range(1, N_STEPS):
        if not np.array_equal(cap.position_ids[s].numpy()[0], pos_self):
            raise RuntimeError(f"step {s}: position_ids changed")

    # ---- rebuild the per-layer residual stream ------------------------------
    hidden_in = np.empty((N_STEPS, N_LAYERS, SEQ, 720), np.float32)
    hidden_out = np.empty_like(hidden_in)
    with torch.no_grad():
        for s in range(N_STEPS):
            for L in range(N_LAYERS):
                hi = cap.hidden_in[(s, L)]
                op = cap.o_proj_out[(s, L)]
                # `out_emb += hidden_states` is an IN-PLACE add on a bf16
                # tensor: torch accumulates in fp32 and rounds to bf16 ONCE, at
                # the store. Pre-downcasting hi and then adding rounds TWICE and
                # is off by up to 0.03 on layer 0 (whose hi is the fp32
                # embed_suffix output); layers 1-15 are unaffected since their
                # hi is already bf16. Verified by the residual-continuity check
                # below, which fails for the pre-downcast variant.
                out = (hi.float() + op.float()).to(op.dtype) + cap.mlp_out[(s, L)]
                hidden_in[s, L] = _f32(hi)[0]
                hidden_out[s, L] = _f32(out)[0]
    # the residual stream must be continuous across layers within a step
    for s in range(N_STEPS):
        for L in range(1, N_LAYERS):
            np.testing.assert_allclose(
                hidden_in[s, L], hidden_out[s, L - 1], rtol=0, atol=0
            )

    def _stack_sl(d, layers, shape):
        a = np.empty((N_STEPS, len(layers), *shape), np.float32)
        for s in range(N_STEPS):
            for j, L in enumerate(layers):
                a[s, j] = _f32(d[(s, L)])[0]
        return a

    all_layers = list(range(N_LAYERS))
    q_rope = _stack_sl(cap.q_rope, all_layers, (SEQ, N_Q_HEADS, HEAD_DIM))
    att_out = _stack_sl(cap.att_out, all_layers, (SEQ, Q_DIM))
    # self layers: only the 50 NEW K/V rows (columns 0..240 == prefix cache)
    k_new_self = np.stack(
        [
            np.stack([_f32(cap.k_full[(s, L)])[0, PREFIX_LEN:] for L in SELF_LAYERS])
            for s in range(N_STEPS)
        ]
    )
    v_new_self = np.stack(
        [
            np.stack([_f32(cap.v_full[(s, L)])[0, PREFIX_LEN:] for L in SELF_LAYERS])
            for s in range(N_STEPS)
        ]
    )
    # cross layers: step-invariant, stored once
    k_cross = np.stack([_f32(cap.k_full[(0, L)])[0] for L in CROSS_LAYERS])
    v_cross = np.stack([_f32(cap.v_full[(0, L)])[0] for L in CROSS_LAYERS])

    suffix_embs = np.stack([_f32(cap.suffix_embs[s])[0] for s in range(N_STEPS)])
    final_norm = np.stack([_f32(cap.final_norm[s])[0] for s in range(N_STEPS)])
    v_t = np.stack([_f32(cap.v_t[s])[0] for s in range(N_STEPS)])
    x_t = np.stack([_f32(cap.x_t[s])[0] for s in range(N_STEPS)])
    timesteps = np.array([cap.timestep[s] for s in range(N_STEPS)], np.float32)
    # x_t_final: x_t after the last update, i.e. the unpadded action chunk
    x_t_final = x_t[-1] + (-1.0 / N_STEPS) * v_t[-1]

    np.savez(
        OUT,
        prefix_pad_masks=prefix_pad_masks,
        mask_self=mask_self,
        mask_cross=mask_cross,
        pos_self=pos_self,
        pos_cross=pos_cross,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
        x_t=x_t,
        x_t_final=x_t_final,
        timesteps=timesteps,
        suffix_embs=suffix_embs,
        hidden_in=hidden_in,
        hidden_out=hidden_out,
        q_rope=q_rope,
        att_out=att_out,
        k_new_self=k_new_self,
        v_new_self=v_new_self,
        k_cross=k_cross,
        v_cross=v_cross,
        final_norm=final_norm,
        v_t=v_t,
        action_chunk=_f32(action_chunk),
        self_layers=np.array(SELF_LAYERS, np.int64),
        cross_layers=np.array(CROSS_LAYERS, np.int64),
    )
    import os

    print(
        f"[expert-oracle] wrote {OUT} "
        f"({os.path.getsize(OUT) / 1e6:.1f} MB): "
        f"hidden_in{hidden_in.shape} q_rope{q_rope.shape} "
        f"k_new_self{k_new_self.shape} k_cross{k_cross.shape} "
        f"final_norm{final_norm.shape} v_t{v_t.shape} "
        f"action_chunk{tuple(action_chunk.shape)}"
    )
    print(
        f"[expert-oracle] prefix real tokens {int(prefix_pad_masks.sum())}/{PREFIX_LEN}, "
        f"self positions {pos_self[0]}..{pos_self[-1]}, "
        f"cross positions {pos_cross[0]}..{pos_cross[-1]}"
    )


if __name__ == "__main__":
    main()
