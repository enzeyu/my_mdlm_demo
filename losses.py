"""Loss helpers for coarse-to-fine training."""

from __future__ import annotations

import numpy as np

from diffusion_utils import softmax


def mse_loss(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[1] != b.shape[1]:
        b = b[:, : a.shape[1], :]
    return float(np.mean((a - b) ** 2))


def kl_distillation_loss(student_logits: np.ndarray, teacher_logits: np.ndarray, mask: np.ndarray, temperature: float = 2.0) -> float:
    idx = np.where(mask.reshape(-1))[0]
    if len(idx) == 0:
        return 0.0
    s = softmax(student_logits.reshape(-1, student_logits.shape[-1])[idx] / temperature, axis=-1)
    t = softmax(teacher_logits.reshape(-1, teacher_logits.shape[-1])[idx] / temperature, axis=-1)
    return float((t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(s, 1e-12)))).sum(axis=-1).mean())
