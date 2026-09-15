"""CPU target-feature fusion and learned drafter context KV, from packed weights.

This module never dispatches an NPU program. Projection weights are decoded from
the same cache as the NPU drafter. Norm parameters and epsilon are taken from
the pinned official checkpoint. Context KV is rounded to BF16 on upload.
"""
import numpy as np

from draft_weights import DraftWeights, PACKED_CACHE
from packed_weights import dequant_cascade, geom_for
from target_config import TargetConfig, load_geometry


def rmsnorm(x, weight, eps):
    x = np.asarray(x, np.float32)
    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) * np.asarray(weight, np.float32)


def rope(x, positions, theta=1_000_000):
    """Half-split RoPE on [position, KV head, dimension], FP32 CPU reference."""
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angle = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    cos = np.cos(angle).astype(np.float32)[:, None, :]
    sin = np.sin(angle).astype(np.float32)[:, None, :]
    left, right = x[..., :half], x[..., half:]
    return np.concatenate((left*cos - right*sin, left*sin + right*cos), axis=-1)


class ContextPrepass:
    def __init__(self, weights=None, packed_cache=PACKED_CACHE):
        self.weights = weights or DraftWeights()
        self.eps = float(self.weights.config['rms_norm_eps'])
        self.theta = float(self.weights.config['rope_theta'])
        fd = load_geometry(TargetConfig(role='drafter', layers=5, causal=False))
        self.hidden_norm = self.weights.tensor('hidden_norm.weight').astype(np.float32)
        self.k_norm = [self.weights.layer_qk_norm(i)[1].astype(np.float32) for i in range(5)]
        self.kv_weights = []
        with np.load(packed_cache) as cache:
            self.fc = dequant_cascade(cache['W_fc'], 2560, 12800, geom_for(2560, 12800, fd))
            all_weights = cache['W']
            g = geom_for(6144, 2560, fd)
            from packed_weights import BLOCK_BF16
            count = g.n_blocks * BLOCK_BF16
            for layer in range(5):
                slab = all_weights[layer*fd.W_LAYER:layer*fd.W_LAYER + count]
                kv = dequant_cascade(slab, 6144, 2560, g, rows=(4096, 6144))
                self.kv_weights.append(kv)

    def features(self, taps):
        taps = np.asarray(taps, np.float32)
        if taps.ndim != 2 or taps.shape[1] != 12800:
            raise ValueError(('Expected concatenated target taps', taps.shape))
        # Preserve the five-tap accumulation order of the previous CPU path.
        fused = np.zeros((len(taps), 2560), np.float32)
        for tap in range(5):
            lo, hi = tap*2560, (tap+1)*2560
            fused += taps[:, lo:hi] @ self.fc[:, lo:hi].T
        return rmsnorm(fused, self.hidden_norm, self.eps)

    def from_features(self, features, positions):
        features = np.asarray(features, np.float32)
        if features.shape != (len(positions), 2560):
            raise ValueError('Features and absolute positions disagree')
        keys, values = [], []
        for layer, weight in enumerate(self.kv_weights):
            kv = features @ weight.T
            k = rmsnorm(kv[:, :1024].reshape(-1, 8, 128), self.k_norm[layer], self.eps)
            keys.append(rope(k, positions, self.theta).reshape(-1, 1024))
            values.append(kv[:, 1024:])
        return np.stack(keys), np.stack(values)

    def run(self, taps, positions):
        features = self.features(taps)
        k, v = self.from_features(features, positions)
        return features, k, v
