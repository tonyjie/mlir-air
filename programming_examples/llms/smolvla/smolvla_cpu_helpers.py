"""CPU NumPy helpers for the SmolVLA backbone port (F32 reference math)."""

import numpy as np


def rms_norm(x, weight, eps=1e-5):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight.astype(np.float32)


def build_prefix_mask(n_prefix, n_state=1, pad_mask=None):
    """Additive attention mask for SmolVLA's prefix-LM pattern.
    Prefix tokens (image+language) attend bidirectionally among themselves but
    NOT to the state token; the state token attends to everything before it.
    Mirrors make_att_2d_masks (modeling_smolvla.py:101-131).

    pad_mask: optional bool array of shape (n_prefix+n_state,), True where the
    token is real and False where it is a tokenizer padding slot (SmolVLA's
    language block is padded to a fixed tokenizer_max_length). Padding tokens
    are excluded from attention both as queries and as keys, mirroring
    make_att_2d_masks' `pad_masks[:,None,:] * pad_masks[:,:,None]` term.
    When omitted (default), no tokens are treated as padding -- this preserves
    the original prefix/state-only masking behavior."""
    n = n_prefix + n_state
    mask = np.zeros((n, n), np.float32)
    # prefix rows cannot see the state column(s)
    mask[:n_prefix, n_prefix:] = -np.inf
    if pad_mask is not None:
        pad_mask = np.asarray(pad_mask, dtype=bool)
        assert pad_mask.shape == (n,), f"pad_mask shape {pad_mask.shape} != {(n,)}"
        invalid = ~pad_mask
        mask[invalid, :] = -np.inf
        mask[:, invalid] = -np.inf
    return mask


def build_padded_mask_and_positions(pad_mask, oracle_len, npu_seq_len=256):
    """Extend the prefix mask + RoPE positions from the real prefix length to
    the NPU kernels' padded sequence length.

    The registry kernels are validated at M=256 while lerobot's prefix is 241
    tokens, so both the additive attention mask and the RoPE position array are
    built at the oracle length and then padded. Padding rows/columns stay -inf
    (fully masked) and padding positions freeze at the last real position via
    `cumsum(pad_mask) - 1`, exactly like SmolVLA's own position_ids.

    Single source of truth shared by the subprocess bridge
    (`run_npu_backbone.py`) and the single-process runtime
    (`smolvla_npu_runtime.BackboneRuntime`)."""
    pad_mask = np.asarray(pad_mask, dtype=bool)
    n_prefix = oracle_len - 1  # 240 visual+language, 1 state token
    mask_o = build_prefix_mask(n_prefix, 1, pad_mask=pad_mask)
    mask_p = np.full((npu_seq_len, npu_seq_len), -np.inf, dtype=np.float32)
    mask_p[:oracle_len, :oracle_len] = mask_o

    pad_mask_p = np.zeros((npu_seq_len,), dtype=bool)
    pad_mask_p[:oracle_len] = pad_mask
    positions_p = np.cumsum(pad_mask_p.astype(np.int64)) - 1
    positions_p = np.clip(positions_p, 0, None)
    return mask_p, positions_p


def _softmax(x, axis=-1):
    """Softmax with HF-eager-attention semantics for fully-masked rows.
    Real SmolVLA masking (smolvlm_with_expert.py eager_attention_forward) uses
    `torch.where(mask, weights, finfo.min)` rather than an additive -inf, so a
    row that is masked out everywhere (e.g. a padding query token, whose whole
    row of pad_2d_masks is False) still gets a well-defined *uniform*
    distribution instead of 0/0 NaN. Reproduce that here for additive -inf
    masks: detect all-(-inf) rows and force uniform probabilities for them."""
    x = np.asarray(x, dtype=np.float32)
    all_masked = np.all(np.isneginf(x), axis=axis, keepdims=True)
    safe_x = np.where(all_masked, 0.0, x)
    m = np.max(safe_x, axis=axis, keepdims=True)
    e = np.exp(safe_x - m)
    p = e / np.sum(e, axis=axis, keepdims=True)
    uniform = np.full_like(x, 1.0 / x.shape[axis])
    return np.where(all_masked, uniform, p)


def noncausal_attention_reference(q, k, v, mask, n_heads, n_kv_heads):
    """q:(L,H,D) k,v:(L,Hkv,D) mask:(L,L) additive. GQA repeat, F32."""
    L, H, D = q.shape
    group = n_heads // n_kv_heads
    out = np.empty((L, H, D), np.float32)
    scale = 1.0 / np.sqrt(D)
    for h in range(H):
        kv = h // group
        s = (q[:, h, :].astype(np.float32) @ k[:, kv, :].astype(np.float32).T) * scale
        s = s + mask
        p = _softmax(s, axis=-1)
        out[:, h, :] = p @ v[:, kv, :].astype(np.float32)
    return out


def _rope_half_split(x, positions, base, head_dim):
    # x:(L,H,D). Non-interleaved (half-split) RoPE.
    half = head_dim // 2
    inv = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    ang = positions[:, None].astype(np.float32) * inv[None, :]  # (L,half)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)[:, None, :]
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], -1)
    return x * cos + rot * sin


def cpu_backbone_forward(prefix_embed, w, cfg, mask, rope_base, positions=None):
    """positions: optional per-token RoPE position array of shape (L,). SmolVLA
    computes `position_ids = cumsum(pad_masks) - 1` (modeling_smolvla.py), so
    padding tokens freeze at the last real position and the state token's
    position is the count of real prefix tokens, NOT its raw sequence index.
    Defaults to plain arange(L) (no padding) when omitted."""
    x = prefix_embed.astype(np.float32)  # (241,960)
    L = x.shape[0]
    pos = np.arange(L) if positions is None else np.asarray(positions)
    for lw in w.layers:
        h = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps)
        q = (h @ lw.wq.astype(np.float32)).reshape(L, cfg.n_heads, cfg.head_dim)
        k = (h @ lw.wk.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ lw.wv.astype(np.float32)).reshape(L, cfg.n_kv_heads, cfg.head_dim)
        q = _rope_half_split(q, pos, rope_base, cfg.head_dim)
        k = _rope_half_split(k, pos, rope_base, cfg.head_dim)
        a = noncausal_attention_reference(q, k, v, mask, cfg.n_heads, cfg.n_kv_heads)
        a = a.reshape(L, cfg.n_heads * cfg.head_dim) @ lw.wo.astype(np.float32)
        x = x + a
        h = rms_norm(x, lw.ffn_norm, cfg.rms_norm_eps)
        g = h @ lw.w_gate.astype(np.float32)
        u = h @ lw.w_up.astype(np.float32)
        silu = g / (1.0 + np.exp(-g))
        x = x + (silu * u) @ lw.w_down.astype(np.float32)
    return rms_norm(x, w.final_norm, cfg.rms_norm_eps)
