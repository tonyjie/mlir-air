"""Pure-CPU self-check of the SmolVLA action-expert reference math.

Replays the expert from the oracle's own inputs (expert_oracle.npz, produced by
expert_prefix.py) using the numpy reference in expert_cpu_helpers.py, and gates
on per-layer cosine. This de-risks the NPU port: it proves the reference math
and the layout conventions (RoPE base/positions, the two mask shapes, the
GQA head mapping, the fp32 cross-layer k/v projection) are right BEFORE any
kernel work. Sibling of test_cpu_backbone.py / test_cpu_helpers.py.

Run with the lerobot venv (needs ml_dtypes + safetensors, not torch):
    /home/jiajli/Projects/smolvla_playground/.venv/bin/python -m pytest test_cpu_expert.py

Note the reference is F32 while the real expert runs bf16, so the gate is
cosine, not exact equality -- the same convention the backbone test uses.
"""

import numpy as np
import pytest

from expert_cpu_helpers import (
    apply_rope,
    embed_suffix,
    expert_forward,
    expert_layer_forward,
    gqa_attention,
)
from expert_weights import SmolVLAExpertConfig, load_expert_weights

ORACLE = "expert_oracle.npz"
COS_GATE = 0.999


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def oracle():
    try:
        return np.load(ORACLE)
    except FileNotFoundError:
        pytest.skip(f"{ORACLE} missing -- run expert_prefix.py in the lerobot venv")


@pytest.fixture(scope="module")
def weights():
    return load_expert_weights()


@pytest.fixture(scope="module")
def cfg():
    return SmolVLAExpertConfig()


# ---------------------------------------------------------------------------
# Unit-level: the two pieces most likely to be silently wrong
# ---------------------------------------------------------------------------


def test_apply_rope_matches_lerobot_form():
    """apply_rope must be a rotation: norms preserved, and position 0 a no-op."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 3, 64)).astype(np.float32)
    y = apply_rope(x, np.zeros(7), 10000.0)
    np.testing.assert_allclose(y, x, atol=1e-6)
    y = apply_rope(x, np.arange(7) + 13, 10000.0)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5
    )


def test_gqa_head_mapping_is_repeat_interleave():
    """q-head h must read kv-head h//3, matching eager_attention_forward's
    expand+reshape (NOT h % n_kv_heads)."""
    rng = np.random.default_rng(1)
    Lq, Lk, H, Hkv, D = 4, 6, 15, 5, 64
    q = rng.standard_normal((Lq, H, D)).astype(np.float32)
    k = rng.standard_normal((Lk, Hkv, D)).astype(np.float32)
    v = rng.standard_normal((Lk, Hkv, D)).astype(np.float32)
    mask = np.ones((Lq, Lk), bool)
    out = gqa_attention(q, k, v, mask, H, Hkv).reshape(Lq, H, D)
    for h in (0, 1, 2, 4, 14):
        kv = h // 3
        s = (q[:, h] @ k[:, kv].T) * D**-0.5
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        np.testing.assert_allclose(out[:, h], p @ v[:, kv], atol=1e-5)


def test_masked_columns_are_ignored():
    """A False mask column must not influence the output at all."""
    rng = np.random.default_rng(2)
    q = rng.standard_normal((3, 15, 64)).astype(np.float32)
    k = rng.standard_normal((8, 5, 64)).astype(np.float32)
    v = rng.standard_normal((8, 5, 64)).astype(np.float32)
    mask = np.ones((3, 8), bool)
    mask[:, 5:] = False
    a = gqa_attention(q, k, v, mask, 15, 5)
    k2, v2 = k.copy(), v.copy()
    k2[5:] = rng.standard_normal((3, 5, 64))
    v2[5:] = rng.standard_normal((3, 5, 64))
    b = gqa_attention(q, k2, v2, mask, 15, 5)
    np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Oracle-level
# ---------------------------------------------------------------------------


def test_oracle_structure(oracle, cfg):
    o = oracle
    assert o["hidden_in"].shape == (10, 16, 50, 720)
    assert o["prefix_k"].shape == (16, 241, 5, 64)
    assert o["k_cross"].shape == (8, 241, 5, 64)
    assert o["mask_self"].shape == (50, 291)
    assert o["mask_cross"].shape == (50, 241)
    assert list(o["self_layers"]) == list(range(0, 16, 2))
    assert list(o["cross_layers"]) == list(range(1, 16, 2))
    # prefix columns of the two masks agree and are NOT all-True (the language
    # block is padded to tokenizer_max_length -- 197 real of 241)
    np.testing.assert_array_equal(o["mask_self"][:, :241], o["mask_cross"])
    assert int(o["prefix_pad_masks"].sum()) == 197
    # suffix block is causal
    np.testing.assert_array_equal(
        o["mask_self"][:, 241:], np.tril(np.ones((50, 50), bool))
    )
    # positions: self is offset by the real-token count, cross is re-based to 0
    np.testing.assert_array_equal(o["pos_self"], 197 + np.arange(50))
    np.testing.assert_array_equal(o["pos_cross"], np.arange(50))


def test_embed_suffix_matches_oracle(oracle, weights, cfg):
    o = oracle
    for s in range(o["timesteps"].shape[0]):
        got = embed_suffix(o["x_t"][s], o["timesteps"][s], weights, cfg)
        c = _cos(got, o["suffix_embs"][s])
        assert c > 0.9999, f"step {s}: embed_suffix cosine {c}"


def test_cross_layer_kv_projection_matches_oracle(oracle, weights, cfg):
    """The fp32 nn.Linear(320,320) re-projection of the backbone cache."""
    o = oracle
    for j, L in enumerate(o["cross_layers"]):
        lw = weights.layers[int(L)]
        kf = o["prefix_k"][L].reshape(241, 320)
        vf = o["prefix_v"][L].reshape(241, 320)
        k = (kf @ lw.wk_cross).reshape(241, 5, 64)
        v = (vf @ lw.wv_cross).reshape(241, 5, 64)
        assert _cos(k, o["k_cross"][j]) > 0.9999, f"L{L} k_cross"
        assert _cos(v, o["v_cross"][j]) > 0.9999, f"L{L} v_cross"


@pytest.mark.parametrize("step", [0, 5, 9])
def test_teacher_forced_per_layer(oracle, weights, cfg, step):
    """Feed each layer the ORACLE's input and check its output -- isolates a
    per-layer error from error accumulated upstream."""
    o = oracle
    self_layers = list(o["self_layers"])
    worst = (1.0, None)
    for L in range(cfg.n_layers):
        out, inter = expert_layer_forward(
            o["hidden_in"][step, L],
            weights.layers[L],
            cfg,
            L,
            o["prefix_k"][L],
            o["prefix_v"][L],
            o["mask_self"],
            o["mask_cross"],
            o["pos_self"],
            o["pos_cross"],
        )
        c_q = _cos(inter["q_rope"], o["q_rope"][step, L])
        c_a = _cos(inter["att_out"], o["att_out"][step, L])
        c_o = _cos(out, o["hidden_out"][step, L])
        # k/v: self layers store only the 50 new rows, cross layers the 241
        if L in self_layers:
            j = self_layers.index(L)
            c_k = _cos(inter["k"][241:], o["k_new_self"][step, j])
            c_v = _cos(inter["v"][241:], o["v_new_self"][step, j])
        else:
            j = list(o["cross_layers"]).index(L)
            c_k = _cos(inter["k"], o["k_cross"][j])
            c_v = _cos(inter["v"], o["v_cross"][j])
        for name, c in (("q", c_q), ("k", c_k), ("v", c_v), ("att", c_a), ("out", c_o)):
            assert c > COS_GATE, f"step {step} layer {L} {name}: cosine {c}"
            if c < worst[0]:
                worst = (c, f"step {step} L{L} {name}")
    print(f"[teacher-forced] step {step}: worst cosine {worst[0]:.6f} at {worst[1]}")


@pytest.mark.parametrize("step", [0, 9])
def test_free_running_step(oracle, weights, cfg, step):
    """Run all 16 layers from the oracle's embed_suffix output with no
    teacher forcing, then check the final norm and v_t."""
    o = oracle
    out, v_t, per_layer = expert_forward(
        o["suffix_embs"][step],
        weights,
        cfg,
        o["prefix_k"],
        o["prefix_v"],
        o["mask_self"],
        o["mask_cross"],
        o["pos_self"],
        o["pos_cross"],
    )
    for L, rec in enumerate(per_layer):
        c = _cos(rec["out"], o["hidden_out"][step, L])
        assert c > COS_GATE, f"step {step} free-running layer {L}: cosine {c}"
    c_norm = _cos(out, o["final_norm"][step])
    c_v = _cos(v_t, o["v_t"][step])
    print(f"[free-running] step {step}: final_norm {c_norm:.6f} v_t {c_v:.6f}")
    assert c_norm > COS_GATE, f"final_norm cosine {c_norm}"
    assert c_v > COS_GATE, f"v_t cosine {c_v}"


def test_full_denoise_loop_reproduces_action_chunk(oracle, weights, cfg):
    """The whole 10-step loop from zero noise, closing on the action chunk."""
    o = oracle
    dt = -1.0 / cfg.num_steps
    x_t = np.zeros((cfg.chunk_size, cfg.action_dim), np.float32)
    for s in range(cfg.num_steps):
        emb = embed_suffix(x_t, o["timesteps"][s], weights, cfg)
        _, v_t, _ = expert_forward(
            emb,
            weights,
            cfg,
            o["prefix_k"],
            o["prefix_v"],
            o["mask_self"],
            o["mask_cross"],
            o["pos_self"],
            o["pos_cross"],
        )
        x_t = x_t + dt * v_t
    n_act = o["action_chunk"].shape[-1]
    c = _cos(x_t[:, :n_act], o["action_chunk"][0])
    print(f"[full-loop] action chunk cosine {c:.6f}")
    assert c > COS_GATE, f"action chunk cosine {c}"
