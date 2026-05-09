"""Pluggable collaboration mechanisms for edge-device training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class CommunicationStats:
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    sync_rounds: int = 0

    @property
    def total_bytes(self) -> int:
        return self.uploaded_bytes + self.downloaded_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


class ErrorDrivenSelector:
    """Select high-uncertainty masked token positions for edge feedback."""

    def __init__(self, top_k_tokens: int, min_confidence: float):
        self.top_k_tokens = int(top_k_tokens)
        self.min_confidence = float(min_confidence)

    @torch.no_grad()
    def select(self, device_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        probs = device_logits.softmax(dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)
        score = entropy + (1.0 - confidence)
        eligible = target_mask & confidence.lt(self.min_confidence)
        if eligible.sum().item() == 0:
            eligible = target_mask
        flat_score = score.masked_fill(~eligible, -1.0).reshape(-1)
        k = min(self.top_k_tokens, int(eligible.sum().item()))
        selected = torch.zeros_like(target_mask)
        if k > 0:
            selected.reshape(-1)[flat_score.topk(k).indices] = True
        return selected


class LogitDistillationCollaboration:
    """Edge-to-device logit distillation on selected token positions."""

    def __init__(self, config: dict):
        collab = config["collaboration"]
        self.sync_interval = int(collab.get("sync_interval", 5))
        self.kd_weight = float(collab.get("kd_weight", 0.5))
        self.temperature = float(collab.get("temperature", 2.0))
        self.selector = ErrorDrivenSelector(
            top_k_tokens=int(collab.get("top_k_tokens", 16)),
            min_confidence=float(collab.get("min_confidence", 0.5)),
        )
        self.comm = CommunicationStats()

    def should_sync(self, step: int) -> bool:
        return step % self.sync_interval == 0

    def distill(
        self,
        step: int,
        device_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self.should_sync(step):
            return device_logits.new_tensor(0.0), {"selected_tokens": 0.0}

        selected = self.selector.select(device_logits.detach(), target_mask)
        selected_count = int(selected.sum().item())
        if selected_count == 0:
            return device_logits.new_tensor(0.0), {"selected_tokens": 0.0}

        temp = self.temperature
        student = F.log_softmax(device_logits[selected] / temp, dim=-1)
        teacher = F.softmax(edge_logits.detach()[selected] / temp, dim=-1)
        kd_loss = F.kl_div(student, teacher, reduction="batchmean") * (temp * temp)

        # Upload selected positions plus corrupted/clean ids as compact int64 metadata.
        self.comm.uploaded_bytes += selected_count * 3 * 8
        # Download dense teacher logits for selected positions as fp32.
        self.comm.downloaded_bytes += selected_count * device_logits.size(-1) * 4
        self.comm.sync_rounds += 1
        return self.kd_weight * kd_loss, {"selected_tokens": float(selected_count)}

