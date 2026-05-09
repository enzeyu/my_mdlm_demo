"""Masked diffusion corruption, losses, and simple denoising sampling."""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn.functional as F


def sample_noise_prob(batch_size: int, config: dict, device: torch.device) -> torch.Tensor:
    low = float(config["diffusion"].get("mask_prob_min", 0.15))
    high = float(config["diffusion"].get("mask_prob_max", 0.65))
    return torch.empty(batch_size, device=device).uniform_(low, high)


def corrupt_tokens(
    clean_ids: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    noise_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask tokens with per-example probabilities; never mask padding."""
    prob = noise_prob[:, None].expand_as(clean_ids)
    mask = (torch.rand_like(clean_ids.float()) < prob) & clean_ids.ne(pad_token_id)
    corrupted = clean_ids.clone()
    corrupted[mask] = mask_token_id
    # Ensure every sequence contributes at least one denoising target.
    for row in range(clean_ids.size(0)):
        if not mask[row].any():
            valid = clean_ids[row].ne(pad_token_id).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                pos = valid[torch.randint(valid.numel(), (1,), device=clean_ids.device)]
                mask[row, pos] = True
                corrupted[row, pos] = mask_token_id
    return corrupted, mask


def denoising_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross entropy on corrupted token positions only."""
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        clean_ids.reshape(-1),
        reduction="none",
    ).view_as(clean_ids)
    denom = target_mask.sum().clamp_min(1)
    return (per_token * target_mask.float()).sum() / denom, per_token.detach()


@torch.no_grad()
def token_recovery_accuracy(logits: torch.Tensor, clean_ids: torch.Tensor, target_mask: torch.Tensor) -> float:
    denom = int(target_mask.sum().item())
    if denom == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    correct = (pred.eq(clean_ids) & target_mask).sum().item()
    return correct / denom


@torch.no_grad()
def evaluate_denoising(model, loader, config: dict, tokenizer, device: torch.device, max_batches: int = 4) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    batches = 0
    for clean_ids in loader:
        clean_ids = clean_ids.to(device)
        noise_prob = sample_noise_prob(clean_ids.size(0), config, device)
        corrupted, target_mask = corrupt_tokens(
            clean_ids,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            noise_prob,
        )
        logits = model(corrupted, noise_prob)
        loss, _ = denoising_loss(logits, clean_ids, target_mask)
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_correct += int((pred.eq(clean_ids) & target_mask).sum().item())
        total_targets += int(target_mask.sum().item())
        batches += 1
        if batches >= max_batches:
            break
    return {
        "val_loss": total_loss / max(batches, 1),
        "token_acc": total_correct / max(total_targets, 1),
    }


@torch.no_grad()
def measure_sampling(model, config: dict, tokenizer, device: torch.device) -> Dict[str, float]:
    """Measure iterative masked-token generation latency from an all-mask prompt."""
    model.eval()
    seq_len = int(config["data"]["seq_len"])
    steps = int(config["diffusion"].get("sampling_steps", 8))
    x = torch.full((1, seq_len), tokenizer.mask_token_id, dtype=torch.long, device=device)
    start = time.perf_counter()
    for idx in range(steps):
        noise_prob = torch.full((1,), max(0.01, 1.0 - idx / max(steps, 1)), device=device)
        logits = model(x, noise_prob)
        probs = logits.softmax(dim=-1)
        confidence, pred = probs.max(dim=-1)
        masked = x.eq(tokenizer.mask_token_id)
        if masked.any():
            scores = confidence.masked_fill(~masked, -1.0)
            reveal = max(1, int(masked.sum().item() / max(steps - idx, 1)))
            flat_pos = scores.view(-1).topk(min(reveal, int(masked.sum().item()))).indices
            x.view(-1)[flat_pos] = pred.view(-1)[flat_pos]
    latency = time.perf_counter() - start
    tokens = seq_len
    return {
        "sampling_latency": latency,
        "tokens_per_sec": tokens / max(latency, 1e-9),
        "denoising_steps": steps,
    }

