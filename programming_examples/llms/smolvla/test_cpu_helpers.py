import numpy as np
from ml_dtypes import bfloat16
from smolvla_cpu_helpers import (
    rms_norm,
    build_prefix_mask,
    noncausal_attention_reference,
)


def test_prefix_mask_shape_and_bidirectional():
    # 240 bidirectional prefix tokens + 1 state token = 241
    m = build_prefix_mask(n_prefix=240, n_state=1)  # additive mask (241,241)
    assert m.shape == (241, 241)
    # prefix block is fully bidirectional -> all zeros in [0:240,0:240]
    assert np.all(m[:240, :240] == 0.0)
    # state token (row 240) can attend to everything before it (no -inf in its row up to 241)
    assert np.all(m[240, :241] == 0.0)
    # prefix tokens CANNOT attend to the state token (col 240) -> -inf
    assert np.all(np.isneginf(m[:240, 240]))


def test_noncausal_attention_matches_manual():
    rng = np.random.default_rng(0)
    L, H, Hkv, D = 8, 2, 1, 4
    q = rng.standard_normal((L, H, D)).astype(np.float32)
    k = rng.standard_normal((L, Hkv, D)).astype(np.float32)
    v = rng.standard_normal((L, Hkv, D)).astype(np.float32)
    mask = np.zeros((L, L), np.float32)  # full bidirectional
    out = noncausal_attention_reference(q, k, v, mask, n_heads=H, n_kv_heads=Hkv)
    assert out.shape == (L, H, D)
    # head 0 vs manual softmax(QK^T/sqrt(D))V with kv-head 0
    s = (q[:, 0, :] @ k[:, 0, :].T) / np.sqrt(D)
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(out[:, 0, :], p @ v[:, 0, :], atol=1e-5)
