"""Device-side coarse masked diffusion denoiser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


@dataclass
class CoarseDeviceConfig:
    coarse_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    seq_len: int
    lr: float
    seed: int = 7


class CoarseDeviceDenoiser:
    """A compact Transformer-style denoiser trained in low-dimensional coarse space."""

    def __init__(self, config: CoarseDeviceConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        h = config.hidden_size
        c = config.coarse_dim
        self.in_proj = rng.normal(0, 1 / max(c, 1) ** 0.5, size=(c, h)).astype(np.float32)
        self.time_proj = rng.normal(0, 0.08, size=(1, h)).astype(np.float32)
        self.pos = rng.normal(0, 0.03, size=(config.seq_len, h)).astype(np.float32)
        self.qkv = [rng.normal(0, 1 / h**0.5, size=(h, h * 3)).astype(np.float32) for _ in range(config.num_layers)]
        self.ff1 = [rng.normal(0, 1 / h**0.5, size=(h, h * 2)).astype(np.float32) for _ in range(config.num_layers)]
        self.ff2 = [rng.normal(0, 1 / (h * 2) ** 0.5, size=(h * 2, h)).astype(np.float32) for _ in range(config.num_layers)]
        self.out = rng.normal(0, 1 / h**0.5, size=(h, c)).astype(np.float32)
        self.bias = np.zeros((c,), dtype=np.float32)

    @property
    def params(self) -> int:
        return int(
            self.in_proj.size
            + self.time_proj.size
            + self.pos.size
            + self.out.size
            + self.bias.size
            + sum(x.size for x in self.qkv + self.ff1 + self.ff2)
        )

    def forward(self, noisy_coarse: np.ndarray, t: np.ndarray, return_hidden: bool = False):
        h = noisy_coarse @ self.in_proj
        h = h + self.pos[: h.shape[1]][None, :, :] + t[:, None, None] * self.time_proj[None, :, :]
        for layer in range(self.config.num_layers):
            residual = h
            attn = self._attention(_layer_norm(h), self.qkv[layer])
            h = residual + attn
            h = h + _gelu(_layer_norm(h) @ self.ff1[layer]) @ self.ff2[layer]
        hidden = _layer_norm(h)
        pred = hidden @ self.out + self.bias
        if return_hidden:
            return pred.astype(np.float32), hidden.astype(np.float32)
        return pred.astype(np.float32)

    def train_step(self, noisy_coarse: np.ndarray, clean_coarse: np.ndarray, t: np.ndarray) -> float:
        pred, hidden = self.forward(noisy_coarse, t, return_hidden=True)
        if pred.shape[1] != clean_coarse.shape[1]:
            clean_coarse = clean_coarse[:, : pred.shape[1], :]
        err = pred - clean_coarse
        loss = float(np.mean(err**2))
        grad = (2.0 / np.prod(err.shape)) * err
        grad_w = hidden.reshape(-1, hidden.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        grad_b = grad.sum(axis=(0, 1))
        clip = 1.0
        grad_w = np.clip(grad_w, -clip, clip)
        grad_b = np.clip(grad_b, -clip, clip)
        self.out -= self.config.lr * grad_w.astype(np.float32)
        self.bias -= self.config.lr * grad_b.astype(np.float32)
        return loss

    def _attention(self, x: np.ndarray, qkv_w: np.ndarray) -> np.ndarray:
        qkv = x @ qkv_w
        q, k, v = np.split(qkv, 3, axis=-1)
        scale = q.shape[-1] ** -0.5
        scores = (q @ np.swapaxes(k, -1, -2)) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
        return weights @ v

    def flops_per_batch(self, batch_size: int, seq_len: int) -> float:
        h = self.config.hidden_size
        c = self.config.coarse_dim
        layers = self.config.num_layers
        return float(batch_size * seq_len * c * h + layers * batch_size * (seq_len * seq_len * h + seq_len * h * h * 6))
