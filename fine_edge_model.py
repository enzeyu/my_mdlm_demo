"""Edge-side fine token masked diffusion refiner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coarse_device_model import _gelu, _layer_norm
from diffusion_utils import cross_entropy_and_grad


@dataclass
class FineEdgeConfig:
    vocab_size: int
    total_vocab_size: int
    seq_len: int
    coarse_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    conditioning: str
    lr: float
    seed: int = 17


class FineEdgeRefiner:
    """Bidirectional Transformer-style masked token refiner conditioned on coarse semantics."""

    def __init__(self, config: FineEdgeConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        h = config.hidden_size
        self.embed = rng.normal(0, 0.04, size=(config.total_vocab_size, h)).astype(np.float32)
        self.pos = rng.normal(0, 0.03, size=(config.seq_len, h)).astype(np.float32)
        self.time_proj = rng.normal(0, 0.08, size=(1, h)).astype(np.float32)
        self.coarse_adapter = rng.normal(0, 1 / max(config.coarse_dim, 1) ** 0.5, size=(config.coarse_dim, h)).astype(np.float32)
        self.qkv = [rng.normal(0, 1 / h**0.5, size=(h, h * 3)).astype(np.float32) for _ in range(config.num_layers)]
        self.ff1 = [rng.normal(0, 1 / h**0.5, size=(h, h * 3)).astype(np.float32) for _ in range(config.num_layers)]
        self.ff2 = [rng.normal(0, 1 / (h * 3) ** 0.5, size=(h * 3, h)).astype(np.float32) for _ in range(config.num_layers)]
        self.lm_head = rng.normal(0, 1 / h**0.5, size=(h, config.vocab_size)).astype(np.float32)
        self.bias = np.zeros((config.vocab_size,), dtype=np.float32)

    @property
    def params(self) -> int:
        return int(
            self.embed.size
            + self.pos.size
            + self.time_proj.size
            + self.coarse_adapter.size
            + self.lm_head.size
            + self.bias.size
            + sum(x.size for x in self.qkv + self.ff1 + self.ff2)
        )

    def forward(self, input_ids: np.ndarray, t: np.ndarray, coarse: np.ndarray | None = None, return_hidden: bool = False):
        safe = np.clip(input_ids, 0, self.config.total_vocab_size - 1)
        h = self.embed[safe] + self.pos[: input_ids.shape[1]][None, :, :] + t[:, None, None] * self.time_proj[None, :, :]
        if coarse is not None:
            cond = self._expand_coarse(coarse, input_ids.shape[1]) @ self.coarse_adapter
            h = h + cond
        for layer in range(self.config.num_layers):
            h = h + self._attention(_layer_norm(h), self.qkv[layer])
            h = h + _gelu(_layer_norm(h) @ self.ff1[layer]) @ self.ff2[layer]
        hidden = _layer_norm(h)
        logits = hidden @ self.lm_head + self.bias
        if return_hidden:
            return logits.astype(np.float32), hidden.astype(np.float32)
        return logits.astype(np.float32)

    def train_step(self, input_ids: np.ndarray, targets: np.ndarray, target_mask: np.ndarray, t: np.ndarray, coarse=None):
        logits, hidden = self.forward(input_ids, t, coarse=coarse, return_hidden=True)
        loss, grad_logits, acc = cross_entropy_and_grad(logits, targets, target_mask)
        grad_w = hidden.reshape(-1, hidden.shape[-1]).T @ grad_logits.reshape(-1, grad_logits.shape[-1])
        grad_b = grad_logits.sum(axis=(0, 1))
        grad_w = np.clip(grad_w, -1.0, 1.0)
        grad_b = np.clip(grad_b, -1.0, 1.0)
        self.lm_head -= self.config.lr * grad_w.astype(np.float32)
        self.bias -= self.config.lr * grad_b.astype(np.float32)
        return loss, acc, logits, hidden

    def _expand_coarse(self, coarse: np.ndarray, seq_len: int) -> np.ndarray:
        if coarse.shape[1] == seq_len:
            return coarse
        repeat = int(np.ceil(seq_len / coarse.shape[1]))
        return np.repeat(coarse, repeat, axis=1)[:, :seq_len, :]

    def _attention(self, x: np.ndarray, qkv_w: np.ndarray) -> np.ndarray:
        qkv = x @ qkv_w
        q, k, v = np.split(qkv, 3, axis=-1)
        scores = (q @ np.swapaxes(k, -1, -2)) / max(q.shape[-1] ** 0.5, 1.0)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
        return weights @ v

    def flops_per_batch(self, batch_size: int, seq_len: int) -> float:
        h = self.config.hidden_size
        layers = self.config.num_layers
        return float(batch_size * seq_len * h * self.config.vocab_size + layers * batch_size * (seq_len * seq_len * h + seq_len * h * h * 8))
