# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Vision-Encoder (SigLIP ViT) CPU reference math.

F32 numpy references for every vision op, plus the two host-side reshape steps
(im2col patch-embed, connector pixel-shuffle). Sibling of
`smolvla_cpu_helpers.py`, but for the SigLIP ViT.

`cpu_vit_forward` is the whole-encoder oracle used to de-risk the reshape/bias/
norm/attention math BEFORE any NPU work: it must match the real lerobot vision
output near-exactly (per-layer cosine > 0.999).

All math mirrors HF `transformers/models/smolvlm/modeling_smolvlm.py`:
  - pre-norm LayerNorm (affine gamma+beta, eps=1e-6)
  - bidirectional MHA, 12 heads, head_dim 64, scale 1/8, softmax in fp32, biases
  - GELU-tanh MLP (fc1 -> gelu_tanh -> fc2, both with bias)
  - patch embed = im2col + linear (stride==kernel==16, non-overlapping)
  - connector = pixel-shuffle (space-to-depth factor 4) + linear (no bias)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Elementwise / norm references
# ---------------------------------------------------------------------------


def layer_norm(x, gamma, beta, eps=1e-6):
    """Affine LayerNorm over the last axis (subtract mean, divide by std).

    HF `nn.LayerNorm`: reductions in fp32, then affine. x: (..., N).
    """
    x = x.astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # population variance (unbiased=False)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * gamma.astype(np.float32) + beta.astype(np.float32)


def gelu_tanh(x):
    """GELU tanh approximation (gelu_pytorch_tanh), matches the A3-2 kernel.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    x = x.astype(np.float32)
    c = np.sqrt(2.0 / np.pi).astype(np.float32)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Attention reference (bidirectional MHA, no mask, no GQA)
# ---------------------------------------------------------------------------


def mha_bidirectional(q, k, v, n_heads, head_dim, scale):
    """Standard bidirectional multi-head attention, no mask, no GQA.

    q, k, v: (seq, n_heads*head_dim). Returns (seq, n_heads*head_dim).
    Softmax in fp32. scale = head_dim^-0.5 applied to the QK^T scores.
    """
    seq = q.shape[0]
    q = q.astype(np.float32).reshape(seq, n_heads, head_dim)
    k = k.astype(np.float32).reshape(seq, n_heads, head_dim)
    v = v.astype(np.float32).reshape(seq, n_heads, head_dim)
    out = np.empty((seq, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        scores = (q[:, h, :] @ k[:, h, :].T) * scale  # (seq, seq)
        probs = _softmax(scores, axis=-1)
        out[:, h, :] = probs @ v[:, h, :]
    return out.reshape(seq, n_heads * head_dim)


# ---------------------------------------------------------------------------
# Host reshape glue (correctness-critical)
# ---------------------------------------------------------------------------


def im2col_patch_embed(pixel_values, patch_w, patch_b, pos_embed, patch_size=16):
    """Patch embedding as im2col + linear + position embedding.

    pixel_values: (3, H, W) with H=W=512. patch_w: (C*ph*pw, out)=(768,768).
    patch_b: (out,). pos_embed: (num_patches, out)=(1024,768).

    Non-overlapping patches (stride==kernel==patch_size). Patch pixels are
    extracted in (c, kh, kw) order to match the HF conv weight reshape
    (out, C, kh, kw) -> (out, C*kh*kw). Token index = ph*grid + pw (row-major).
    For a full 512x512 image the position_ids collapse to arange(1024), so the
    full pos_embed matrix is added directly.
    """
    C, H, W = pixel_values.shape
    grid = H // patch_size  # 32
    num_patches = grid * grid  # 1024
    x = pixel_values.astype(np.float32)
    cols = np.empty((num_patches, C * patch_size * patch_size), dtype=np.float32)
    for ph in range(grid):
        for pw in range(grid):
            patch = x[
                :,
                ph * patch_size : (ph + 1) * patch_size,
                pw * patch_size : (pw + 1) * patch_size,
            ]  # (C, ph, pw)
            cols[ph * grid + pw] = patch.reshape(-1)  # (c, kh, kw) order
    out = cols @ patch_w.astype(np.float32) + patch_b.astype(np.float32)
    return out + pos_embed.astype(np.float32)  # (1024, 768)


def pixel_shuffle(x, scale_factor=4):
    """Connector pixel-shuffle (space-to-depth), verified bit-exact vs HF.

    x: (num_patches, emb) = (1024, 768). Returns (64, 12288).
    1024 = 32x32 spatial; factor 4 -> 8x8 = 64 tokens, 768*16 = 12288 channels.
    """
    x = x.astype(np.float32)
    n, emb = x.shape
    grid = int(round(np.sqrt(n)))  # 32
    s = scale_factor
    h2 = grid // s  # 8
    x = x.reshape(grid, grid, emb)  # (h, w, c)
    x = x.reshape(h2, s, h2, s, emb)  # (h2, dh, w2, dw, c)
    x = x.transpose(0, 2, 1, 3, 4)  # (h2, w2, dh, dw, c)
    return x.reshape(h2 * h2, s * s * emb)  # (64, 12288)


# ---------------------------------------------------------------------------
# Whole-encoder oracle
# ---------------------------------------------------------------------------


def cpu_vit_layer(x, lw, cfg):
    """One SigLIP encoder layer (pre-norm), F32 reference.

    x: (seq, emb). lw: VisionLayerWeights. Returns (seq, emb).
    """
    # --- attention block ---
    h = layer_norm(x, lw.ln1_w, lw.ln1_b, cfg.layer_norm_eps)
    q = h @ lw.wq.astype(np.float32) + lw.bq.astype(np.float32)
    k = h @ lw.wk.astype(np.float32) + lw.bk.astype(np.float32)
    v = h @ lw.wv.astype(np.float32) + lw.bv.astype(np.float32)
    attn = mha_bidirectional(q, k, v, cfg.n_heads, cfg.head_dim, cfg.attn_scale)
    attn = attn @ lw.wo.astype(np.float32) + lw.bo.astype(np.float32)
    x = x.astype(np.float32) + attn
    # --- MLP block ---
    h = layer_norm(x, lw.ln2_w, lw.ln2_b, cfg.layer_norm_eps)
    h = h @ lw.w_fc1.astype(np.float32) + lw.b_fc1.astype(np.float32)
    h = gelu_tanh(h)
    h = h @ lw.w_fc2.astype(np.float32) + lw.b_fc2.astype(np.float32)
    return x + h


def cpu_vit_forward(
    pixel_values, weights, cfg, return_per_layer=False, do_connector=True
):
    """Full SigLIP ViT + optional connector, F32 reference (the oracle).

    pixel_values: (3, 512, 512). Returns a dict with:
        patch_embed: (1024, 768)
        layer_hidden: list of 12 (1024, 768) [if return_per_layer]
        post_ln: (1024, 768)
        connector: (64, 960) [if do_connector]
    """
    x = im2col_patch_embed(
        pixel_values,
        weights.patch_w,
        weights.patch_b,
        weights.pos_embed,
        cfg.patch_size,
    )
    per_layer = []
    for lw in weights.layers:
        x = cpu_vit_layer(x, lw, cfg)
        if return_per_layer:
            per_layer.append(x.copy())
    post_ln = layer_norm(x, weights.post_ln_w, weights.post_ln_b, cfg.layer_norm_eps)
    result = {"patch_embed": None, "post_ln": post_ln}
    if return_per_layer:
        result["layer_hidden"] = per_layer
    if do_connector:
        shuffled = pixel_shuffle(post_ln)  # (64, 12288)
        result["connector"] = shuffled @ weights.connector_w.astype(
            np.float32
        )  # (64, 960)
    return result
