"""CPU NumPy reference math for the SmolVLA action-expert (F32).

Sibling of `smolvla_cpu_helpers.py` (backbone). Three things differ enough from
the backbone reference that they cannot be reused as-is:

1. RoPE convention. The backbone helper's `_rope_half_split` uses the HF
   inverse-frequency form `base ** (-2i/D)`. SmolVLA's own `apply_rope`
   (smolvlm_with_expert.py) instead computes
   `timescale = base ** ((2/D) * arange(D/2))` and divides -- algebraically the
   same angles, but it is written out here in apply_rope's own form so a future
   divergence is visible at the source, and because the expert applies RoPE at
   two different position bases (self: 197..246, cross: 0..49).

2. Boolean masks, not additive. `eager_attention_forward` does
   `where(mask, weights, finfo(f32).min)` -- so this module takes bool masks
   directly (True = attend), matching what the oracle stores.

3. q_dim (960) != emb_dim (720), so the attention output must be reshaped to
   960 before o_proj, and the residual add happens at 720.
"""

import numpy as np

BIG_NEG = np.finfo(np.float32).min


def rms_norm(x, weight, eps=1e-5):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight.astype(np.float32)


def apply_rope(x, positions, base=10000.0):
    """x:(L,H,D) -> (L,H,D). Transcribes smolvlm_with_expert.apply_rope.

    Half-split (non-interleaved): with x = [x1 | x2],
    out = [x1*cos - x2*sin | x2*cos + x1*sin].
    """
    x = np.asarray(x, np.float32)
    d = x.shape[-1]
    half = d // 2
    freq_exponents = (2.0 / d) * np.arange(half, dtype=np.float32)
    timescale = base**freq_exponents
    radians = np.asarray(positions, np.float32)[:, None] / timescale[None, :]
    cos = np.cos(radians)[:, None, :]  # (L,1,half)
    sin = np.sin(radians)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def gqa_attention(q, k, v, mask, n_heads, n_kv_heads):
    """q:(Lq,H,D)  k,v:(Lk,Hkv,D)  mask:(Lq,Lk) bool (True = attend).
    Returns (Lq, H*D) -- flattened, ready for o_proj. F32 throughout, which is
    what eager_attention_forward does (it upcasts q/k to f32 explicitly)."""
    q = np.asarray(q, np.float32)
    k = np.asarray(k, np.float32)
    v = np.asarray(v, np.float32)
    Lq, H, D = q.shape
    group = n_heads // n_kv_heads
    scale = D**-0.5
    out = np.empty((Lq, H, D), np.float32)
    for h in range(H):
        kv = h // group  # eager_attention_forward expands kv heads with
        # repeat_interleave semantics (`[:,:,:,None,:].expand(...).reshape`),
        # so q-head h maps to kv-head h//group, not h % n_kv_heads.
        s = (q[:, h, :] @ k[:, kv, :].T) * scale
        s = np.where(mask, s, BIG_NEG)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, h, :] = p @ v[:, kv, :]
    return out.reshape(Lq, H * D)


def silu(x):
    return x / (1.0 + np.exp(-x))


def create_sinusoidal_pos_embedding(t, dim, min_period, max_period):
    """Transcribes modeling_smolvla.create_sinusoidal_pos_embedding for a
    scalar timestep. lerobot builds it in float64 (get_safe_dtype on CPU), so
    do the same before casting down."""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * np.pi
    sin_input = scaling * float(t)
    return np.concatenate([np.sin(sin_input), np.cos(sin_input)]).astype(np.float32)


def embed_suffix(x_t, timestep, w, cfg, min_period=0.004, max_period=4.0):
    """Reference for VLAFlowMatching.embed_suffix. x_t:(50,32) -> (50,720)."""
    action_emb = x_t.astype(np.float32) @ w.action_in_w + w.action_in_b
    time_emb = create_sinusoidal_pos_embedding(
        timestep, cfg.emb_dim, min_period, max_period
    )
    time_emb = np.broadcast_to(time_emb, action_emb.shape)
    h = np.concatenate([action_emb, time_emb], axis=-1)  # (50,1440)
    h = h @ w.action_time_in_w + w.action_time_in_b
    h = silu(h)
    return h @ w.action_time_out_w + w.action_time_out_b


def expert_layer_forward(
    x,
    lw,
    cfg,
    layer_idx,
    prefix_k,
    prefix_v,
    mask_self,
    mask_cross,
    pos_self,
    pos_cross,
):
    """One expert layer. x:(50,720) -> (50,720).

    prefix_k/prefix_v: this layer's backbone KV cache, (241,5,64). On SELF
    (even) layers prefix_k is already RoPE'd at prefix positions (the backbone
    stored it post-RoPE) and is used verbatim as the first 241 K columns; the 50
    new action-token K get RoPE at pos_self. On CROSS (odd) layers the cache is
    flattened to (241,320), pushed through the fp32 wk_cross/wv_cross, and gets
    NO RoPE at all -- only q is RoPE'd, at the re-based pos_cross.

    Returns (out, intermediates) where intermediates carries q_rope/k/v/att_out
    for sub-op-level comparison against the oracle.
    """
    x = np.asarray(x, np.float32)
    L = x.shape[0]
    h = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps)
    q = (h @ lw.wq.astype(np.float32)).reshape(L, cfg.n_heads, cfg.head_dim)

    if cfg.is_self_attn(layer_idx):
        k_new = (h @ lw.wk.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        v_new = (h @ lw.wv.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        q = apply_rope(q, pos_self, cfg.rope_base)
        k_new = apply_rope(k_new, pos_self, cfg.rope_base)
        k = np.concatenate([prefix_k.astype(np.float32), k_new], axis=0)
        v = np.concatenate([prefix_v.astype(np.float32), v_new], axis=0)
        mask = mask_self
    else:
        n_p = prefix_k.shape[0]
        kf = prefix_k.astype(np.float32).reshape(n_p, cfg.kv_dim)
        vf = prefix_v.astype(np.float32).reshape(n_p, cfg.kv_dim)
        k = (kf @ lw.wk_cross).reshape(n_p, cfg.n_kv_heads, cfg.head_dim)
        v = (vf @ lw.wv_cross).reshape(n_p, cfg.n_kv_heads, cfg.head_dim)
        q = apply_rope(q, pos_cross, cfg.rope_base)
        mask = mask_cross

    att = gqa_attention(q, k, v, mask, cfg.n_heads, cfg.n_kv_heads)  # (50,960)
    x = x + att @ lw.wo.astype(np.float32)
    h = rms_norm(x, lw.ffn_norm, cfg.rms_norm_eps)
    g = h @ lw.w_gate.astype(np.float32)
    u = h @ lw.w_up.astype(np.float32)
    x = x + (silu(g) * u) @ lw.w_down.astype(np.float32)
    return x, {"q_rope": q, "k": k, "v": v, "att_out": att}


def expert_forward(
    suffix_emb, w, cfg, prefix_k, prefix_v, mask_self, mask_cross, pos_self, pos_cross
):
    """Full 16-layer expert for one denoise step. suffix_emb:(50,720).
    Returns (final_norm_out, v_t, per_layer) where per_layer[i] is
    {'in','out', 'q_rope','k','v','att_out'}."""
    x = np.asarray(suffix_emb, np.float32)
    per_layer = []
    for i, lw in enumerate(w.layers):
        rec = {"in": x}
        x, inter = expert_layer_forward(
            x,
            lw,
            cfg,
            i,
            prefix_k[i],
            prefix_v[i],
            mask_self,
            mask_cross,
            pos_self,
            pos_cross,
        )
        rec.update(inter)
        rec["out"] = x
        per_layer.append(rec)
    out = rms_norm(x, w.final_norm, cfg.rms_norm_eps)
    v_t = out @ w.action_out_w + w.action_out_b
    return out, v_t, per_layer
