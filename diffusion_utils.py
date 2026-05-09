"""Masked diffusion corruption, sampling, and text metrics."""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    probs = softmax(x, axis=axis)
    return np.log(np.maximum(probs, 1e-12))


def sample_noise_prob(batch_size: int, config: dict, rng: np.random.Generator) -> np.ndarray:
    low = float(config["diffusion"].get("mask_prob_min", 0.15))
    high = float(config["diffusion"].get("mask_prob_max", 0.65))
    return rng.uniform(low, high, size=(batch_size,)).astype(np.float32)


def corrupt_tokens(
    clean_ids: np.ndarray,
    mask_token_id: int,
    pad_token_id: int,
    noise_prob: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    prob = noise_prob[:, None]
    target_mask = (rng.random(clean_ids.shape) < prob) & (clean_ids != pad_token_id)
    corrupted = clean_ids.copy()
    corrupted[target_mask] = mask_token_id
    for row in range(clean_ids.shape[0]):
        if not target_mask[row].any():
            valid = np.where(clean_ids[row] != pad_token_id)[0]
            if len(valid):
                pos = rng.choice(valid)
                target_mask[row, pos] = True
                corrupted[row, pos] = mask_token_id
    return corrupted, target_mask


def cross_entropy_and_grad(logits: np.ndarray, targets: np.ndarray, target_mask: np.ndarray):
    probs = softmax(logits, axis=-1)
    flat_mask = target_mask.reshape(-1)
    flat_targets = targets.reshape(-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    idx = np.where(flat_mask)[0]
    if len(idx) == 0:
        return 0.0, np.zeros_like(logits), 0.0
    chosen = flat_probs[idx, flat_targets[idx]]
    loss = float(-np.log(np.maximum(chosen, 1e-12)).mean())
    grad = np.zeros_like(flat_probs)
    grad[idx] = flat_probs[idx]
    grad[idx, flat_targets[idx]] -= 1.0
    grad[idx] /= len(idx)
    pred = flat_probs[idx].argmax(axis=-1)
    acc = float((pred == flat_targets[idx]).mean())
    return loss, grad.reshape(logits.shape), acc


def token_recovery_accuracy(logits: np.ndarray, clean_ids: np.ndarray, target_mask: np.ndarray) -> float:
    denom = int(target_mask.sum())
    if denom == 0:
        return 0.0
    pred = logits.argmax(axis=-1)
    return float(((pred == clean_ids) & target_mask).sum() / denom)


def perplexity_surrogate(loss: float) -> float:
    return float(math.exp(min(20.0, max(0.0, loss))))


def iterative_sample(
    forward_fn,
    seq_len: int,
    mask_token_id: int,
    steps: int,
    rng: np.random.Generator,
    temperature: float = 0.9,
) -> Tuple[np.ndarray, float]:
    x = np.full((1, seq_len), mask_token_id, dtype=np.int64)
    start = time.perf_counter()
    for step in range(max(1, steps)):
        t = np.asarray([1.0 - step / max(steps, 1)], dtype=np.float32)
        logits = forward_fn(x, t)
        probs = softmax(logits / max(temperature, 1e-4), axis=-1)
        confidence = probs.max(axis=-1)
        pred = probs.argmax(axis=-1)
        masked = x == mask_token_id
        if not masked.any():
            break
        reveal = max(1, int(masked.sum() / max(steps - step, 1)))
        scores = np.where(masked, confidence, -1.0).reshape(-1)
        pos = np.argpartition(-scores, min(reveal, len(scores) - 1))[:reveal]
        x.reshape(-1)[pos] = pred.reshape(-1)[pos]
    return x[0], time.perf_counter() - start


def distinct_and_repetition(ids: np.ndarray) -> Dict[str, float]:
    toks = [int(x) for x in ids.tolist()]
    if not toks:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_2": 0.0, "repetition_3": 0.0}

    def distinct(n: int) -> float:
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        return len(set(grams)) / max(len(grams), 1)

    def repetition(n: int) -> float:
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        counts = Counter(grams)
        return sum(1 for c in counts.values() if c > 1) / max(len(counts), 1)

    return {
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "repetition_2": repetition(2),
        "repetition_3": repetition(3),
    }
