"""Coarse semantic space construction for edge-device collaboration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class CoarseSpaceConfig:
    vocab_size: int
    total_vocab_size: int
    token_dim: int
    coarse_dim: int
    method: str = "linear"
    segment_size: int = 2
    seed: int = 7


class CoarseSemanticSpace:
    """Compress token embeddings into a lower-dimensional coarse semantic space."""

    def __init__(self, config: CoarseSpaceConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.token_embed = (rng.normal(0, 0.05, size=(config.total_vocab_size, config.token_dim))).astype(np.float32)
        self.proj = (rng.normal(0, 1.0 / max(config.token_dim, 1) ** 0.5, size=(config.token_dim, config.coarse_dim))).astype(
            np.float32
        )
        self.up_proj = (rng.normal(0, 1.0 / max(config.coarse_dim, 1) ** 0.5, size=(config.coarse_dim, config.token_dim))).astype(
            np.float32
        )
        self.vq_codebook = None

    @property
    def params(self) -> int:
        total = self.token_embed.size + self.proj.size + self.up_proj.size
        if self.vq_codebook is not None:
            total += self.vq_codebook.size
        return int(total)

    def token_embeddings(self, ids: np.ndarray) -> np.ndarray:
        safe = np.clip(ids, 0, self.config.total_vocab_size - 1)
        return self.token_embed[safe]

    def encode(self, ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        token_hidden = self.token_embeddings(ids)
        coarse = token_hidden @ self.proj
        if self.config.method == "pooling":
            coarse = self._segment_pool(coarse)
        elif self.config.method == "vq":
            coarse = self._vector_quantize_placeholder(coarse)
        elif self.config.method != "linear":
            raise ValueError(f"Unknown compression method: {self.config.method}")
        stats = self.stats(ids.shape[1], coarse.shape[1])
        return coarse.astype(np.float32), stats

    def expand_to_tokens(self, coarse: np.ndarray, seq_len: int) -> np.ndarray:
        if coarse.shape[1] == seq_len:
            return coarse
        repeat = int(np.ceil(seq_len / coarse.shape[1]))
        return np.repeat(coarse, repeat, axis=1)[:, :seq_len, :]

    def coarse_to_token_logits(self, coarse: np.ndarray, seq_len: int | None = None) -> np.ndarray:
        if seq_len is not None:
            coarse = self.expand_to_tokens(coarse, seq_len)
        token_space = coarse @ self.up_proj
        prototypes = self.token_embed[: self.config.vocab_size]
        return token_space @ prototypes.T

    def alignment_loss(self, pred: np.ndarray, clean: np.ndarray) -> float:
        if pred.shape[1] != clean.shape[1]:
            pred = self.expand_to_tokens(pred, clean.shape[1])
        return float(np.mean((pred - clean) ** 2))

    def stats(self, seq_len: int, coarse_len: int) -> Dict[str, float]:
        fine_bytes = seq_len * self.config.token_dim * 4
        coarse_bytes = coarse_len * self.config.coarse_dim * 4
        return {
            "coarse_len": float(coarse_len),
            "fine_hidden_bytes_per_seq": float(fine_bytes),
            "coarse_bytes_per_seq": float(coarse_bytes),
            "compression_ratio": float(fine_bytes / max(coarse_bytes, 1)),
        }

    def _segment_pool(self, x: np.ndarray) -> np.ndarray:
        seg = max(1, int(self.config.segment_size))
        bsz, seq_len, dim = x.shape
        pad = (-seq_len) % seg
        if pad:
            x = np.concatenate([x, np.zeros((bsz, pad, dim), dtype=x.dtype)], axis=1)
        return x.reshape(bsz, -1, seg, dim).mean(axis=2)

    def _vector_quantize_placeholder(self, x: np.ndarray) -> np.ndarray:
        if self.vq_codebook is None:
            rng = np.random.default_rng(self.config.seed + 97)
            self.vq_codebook = rng.normal(0, 0.1, size=(64, self.config.coarse_dim)).astype(np.float32)
        flat = x.reshape(-1, x.shape[-1])
        dist = ((flat[:, None, :] - self.vq_codebook[None, :, :]) ** 2).sum(axis=-1)
        nearest = self.vq_codebook[dist.argmin(axis=-1)]
        return nearest.reshape(x.shape)
