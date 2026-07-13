"""CPU NumPy helpers for the SmolVLA backbone port (F32 reference math)."""

import numpy as np


def rms_norm(x, weight, eps=1e-5):
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight.astype(np.float32)


def build_prefix_mask(n_prefix, n_state=1):
    """Additive attention mask for SmolVLA's prefix-LM pattern.
    Prefix tokens (image+language) attend bidirectionally among themselves but
    NOT to the state token; the state token attends to everything before it.
    Mirrors make_att_2d_masks (modeling_smolvla.py:101-131)."""
    n = n_prefix + n_state
    mask = np.zeros((n, n), np.float32)
    # prefix rows cannot see the state column(s)
    mask[:n_prefix, n_prefix:] = -np.inf
    return mask


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


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
