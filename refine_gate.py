"""Learned accept gate utilities for GPT-2 draft + MDLM candidate reranking."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = [
    "gpt2_confidence",
    "gpt2_entropy",
    "gpt2_margin",
    "mdlm_confidence",
    "mdlm_entropy",
    "mdlm_margin",
    "score_gap",
    "gpt2_mdlm_agree",
    "gpt2_candidate_rank",
    "refine_ratio",
    "block_uncertainty",
]


class AcceptGateMLP(nn.Module):
    """Small MLP that predicts whether to accept the reranked candidate."""

    def __init__(self, input_size: int = len(FEATURE_NAMES), hidden_size: int = 64, layers: int = 2):
        super().__init__()
        modules: list[nn.Module] = []
        current = input_size
        for _ in range(max(1, int(layers))):
            modules.append(nn.Linear(current, int(hidden_size)))
            modules.append(nn.ReLU())
            current = int(hidden_size)
        modules.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def build_gate(config: dict) -> AcceptGateMLP:
    return AcceptGateMLP(
        input_size=len(FEATURE_NAMES),
        hidden_size=int(config.get("gate_hidden_dim", config.get("gate_hidden_size", 64))),
        layers=int(config.get("gate_layers", 2)),
    )


def pad_gpt2_logits(gpt2_logits: torch.Tensor, target_vocab: int) -> torch.Tensor:
    if gpt2_logits.size(-1) == target_vocab:
        return gpt2_logits
    padded = gpt2_logits.new_full((*gpt2_logits.shape[:-1], target_vocab), -1e4)
    padded[..., : gpt2_logits.size(-1)] = gpt2_logits
    return padded


@torch.no_grad()
def candidate_rerank_features(
    gpt2_logits: torch.Tensor,
    mdlm_logits: torch.Tensor,
    labels: torch.Tensor,
    refine_mask: torch.Tensor,
    candidate_top_k: int,
    lambda_gpt2: float,
    lambda_mdlm: float,
    refine_ratio: float | None = None,
    block_uncertainty: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute reranked candidates, gate features, and supervised labels."""
    gpt2_padded = pad_gpt2_logits(gpt2_logits, mdlm_logits.size(-1))
    top_k = min(int(candidate_top_k), gpt2_logits.size(-1))

    gpt2_log_probs = torch.log_softmax(gpt2_padded.float(), dim=-1)
    mdlm_log_probs = torch.log_softmax(mdlm_logits.float(), dim=-1)
    gpt2_probs = torch.softmax(gpt2_padded.float(), dim=-1)
    mdlm_probs = torch.softmax(mdlm_logits.float(), dim=-1)

    gpt2_top2 = gpt2_probs.topk(2, dim=-1)
    gpt2_top = gpt2_probs.max(dim=-1)
    mdlm_top2 = mdlm_probs.topk(2, dim=-1)
    entropy_denom = max(float(torch.log(torch.tensor(float(gpt2_logits.size(-1)), device=gpt2_logits.device)).item()), 1.0)
    gpt2_entropy = -(gpt2_probs * gpt2_log_probs).sum(dim=-1) / entropy_denom
    mdlm_entropy_denom = max(float(torch.log(torch.tensor(float(mdlm_logits.size(-1)), device=mdlm_logits.device)).item()), 1.0)
    mdlm_entropy = -(mdlm_probs * mdlm_log_probs).sum(dim=-1) / mdlm_entropy_denom

    candidate_ids = gpt2_padded.topk(top_k, dim=-1).indices
    cand_gpt2 = gpt2_log_probs.gather(-1, candidate_ids)
    cand_mdlm = mdlm_log_probs.gather(-1, candidate_ids)
    rerank_scores = float(lambda_gpt2) * cand_gpt2 + float(lambda_mdlm) * cand_mdlm
    rerank_top2 = rerank_scores.topk(min(2, top_k), dim=-1)
    best_rank = rerank_scores.argmax(dim=-1, keepdim=True)
    rerank_token = candidate_ids.gather(-1, best_rank).squeeze(-1)
    rerank_rank_norm = best_rank.squeeze(-1).float() / max(top_k - 1, 1)
    score_gap = rerank_top2.values[..., 0] - rerank_top2.values[..., -1]

    chosen_gpt2_logprob = cand_gpt2.gather(-1, best_rank).squeeze(-1)
    chosen_mdlm_logprob = cand_mdlm.gather(-1, best_rank).squeeze(-1)
    gpt2_pred = gpt2_top.indices
    ratio_feature = torch.full_like(gpt2_top.values, float(refine_ratio if refine_ratio is not None else 0.0))
    uncertainty_feature = block_uncertainty.float() if block_uncertainty is not None else gpt2_entropy

    features = torch.stack(
        [
            gpt2_top.values,
            gpt2_entropy,
            gpt2_top2.values[..., 0] - gpt2_top2.values[..., 1],
            mdlm_top2.values[..., 0],
            mdlm_entropy,
            mdlm_top2.values[..., 0] - mdlm_top2.values[..., 1],
            chosen_mdlm_logprob - chosen_gpt2_logprob,
            gpt2_pred.eq(rerank_token).float(),
            rerank_rank_norm,
            ratio_feature,
            uncertainty_feature,
        ],
        dim=-1,
    )

    gpt2_correct = gpt2_pred.eq(labels)
    rerank_correct = rerank_token.eq(labels)
    trainable_mask = refine_mask & gpt2_correct.ne(rerank_correct)
    accept_label = (rerank_correct & ~gpt2_correct).float()
    candidate_hit_mask = candidate_ids.eq(labels.unsqueeze(-1)).any(dim=-1)

    return {
        "features": features,
        "rerank_token": rerank_token,
        "gpt2_pred": gpt2_pred,
        "trainable_mask": trainable_mask,
        "accept_label": accept_label,
        "candidate_hit_mask": candidate_hit_mask,
        "top_k": torch.tensor(top_k, device=gpt2_logits.device),
    }


def load_gate_checkpoint(path: str, config: dict, device: torch.device) -> AcceptGateMLP:
    payload = torch.load(path, map_location=device)
    gate_config = payload.get("config", config) if isinstance(payload, dict) else config
    gate = build_gate(gate_config).to(device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    gate.load_state_dict(state)
    gate.eval()
    return gate
