"""PyTorch Transformer models for coarse-to-fine masked diffusion LM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration shared by the device and edge Transformer models."""

    vocab_size: int
    max_length: int
    coarse_dim: int
    device_hidden_size: int
    edge_hidden_size: int
    device_layers: int
    edge_layers: int
    device_heads: int
    edge_heads: int
    dropout: float
    pad_token_id: int


class DeviceCoarseModel(nn.Module):
    """Small masked diffusion Transformer that emits coarse states and token logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.device_hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_length, config.device_hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.device_hidden_size),
            nn.SiLU(),
            nn.Linear(config.device_hidden_size, config.device_hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.device_hidden_size,
            nhead=config.device_heads,
            dim_feedforward=config.device_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.device_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.device_hidden_size)

        # This is the explicit coarse semantic bottleneck:
        # token hidden states -> low-dimensional coarse representation.
        self.coarse_proj = nn.Linear(config.device_hidden_size, config.coarse_dim)
        self.device_lm_head = nn.Linear(config.device_hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor):
        """Return device token logits and low-dimensional coarse representations."""
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        h = h + self.time_embed(timesteps[:, None].float()).unsqueeze(1)
        padding_mask = input_ids.eq(self.config.pad_token_id)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)
        coarse = self.coarse_proj(h)
        logits = self.device_lm_head(h)
        return logits, coarse


class EdgeRefineModel(nn.Module):
    """Larger masked diffusion Transformer that refines tokens with optional coarse conditioning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.edge_hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_length, config.edge_hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.edge_hidden_size),
            nn.SiLU(),
            nn.Linear(config.edge_hidden_size, config.edge_hidden_size),
        )
        self.coarse_to_edge = nn.Linear(config.coarse_dim, config.edge_hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.edge_hidden_size,
            nhead=config.edge_heads,
            dim_feedforward=config.edge_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.edge_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.edge_hidden_size)
        self.lm_head = nn.Linear(config.edge_hidden_size, config.vocab_size)

        # Maps edge hidden states into the same coarse space for alignment loss.
        self.edge_to_coarse = nn.Linear(config.edge_hidden_size, config.coarse_dim)

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor, coarse: torch.Tensor | None = None):
        """Return token logits, edge hidden states, and projected coarse targets."""
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        h = h + self.time_embed(timesteps[:, None].float()).unsqueeze(1)
        if coarse is not None:
            h = h + self.coarse_to_edge(coarse)
        padding_mask = input_ids.eq(self.config.pad_token_id)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)
        logits = self.lm_head(h)
        edge_coarse = self.edge_to_coarse(h)
        return logits, h, edge_coarse


class CoarseToFineModel(nn.Module):
    """Container that runs device and edge models for all three evaluation modes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device_model = DeviceCoarseModel(config)
        self.edge_model = EdgeRefineModel(config)

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor, mode: str = "coarse_to_fine"):
        """Run `device_only`, `edge_only`, or `coarse_to_fine` forward pass."""
        device_logits, coarse = self.device_model(input_ids, timesteps)
        if mode == "device_only":
            return {"logits": device_logits, "device_logits": device_logits, "coarse": coarse}
        if mode == "edge_only":
            edge_logits, edge_hidden, edge_coarse = self.edge_model(input_ids, timesteps, coarse=None)
        elif mode == "coarse_to_fine":
            edge_logits, edge_hidden, edge_coarse = self.edge_model(input_ids, timesteps, coarse=coarse)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return {
            "logits": edge_logits,
            "device_logits": device_logits,
            "coarse": coarse,
            "edge_hidden": edge_hidden,
            "edge_coarse": edge_coarse,
        }


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, target_mask: torch.Tensor):
    """Compute cross entropy and accuracy only on masked denoising targets."""
    vocab = logits.size(-1)
    if target_mask.sum().item() == 0:
        return logits.new_tensor(0.0), logits.new_tensor(0.0)
    selected_logits = logits[target_mask]
    selected_labels = labels[target_mask]
    loss = F.cross_entropy(selected_logits.view(-1, vocab), selected_labels.view(-1))
    acc = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
    return loss, acc


def build_model_from_config(config: dict, vocab_size: int, pad_token_id: int) -> CoarseToFineModel:
    """Create the full model from YAML values and tokenizer metadata."""
    model_config = ModelConfig(
        vocab_size=vocab_size,
        max_length=int(config["max_length"]),
        coarse_dim=int(config["coarse_dim"]),
        device_hidden_size=int(config["device_hidden_size"]),
        edge_hidden_size=int(config["edge_hidden_size"]),
        device_layers=int(config["device_layers"]),
        edge_layers=int(config["edge_layers"]),
        device_heads=int(config.get("device_heads", 4)),
        edge_heads=int(config.get("edge_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
        pad_token_id=pad_token_id,
    )
    return CoarseToFineModel(model_config)


def coarse_alignment_loss(device_coarse: torch.Tensor, edge_coarse: torch.Tensor, target_mask: torch.Tensor):
    """Align device coarse states with edge hidden states projected to coarse space."""
    if target_mask.sum().item() == 0:
        return device_coarse.new_tensor(0.0)
    return F.mse_loss(device_coarse[target_mask], edge_coarse.detach()[target_mask])


def coarse_comm_mb(batch_size: int, seq_len: int, coarse_dim: int, dtype_bytes: int = 2) -> float:
    """Estimate coarse representation transfer size in MiB for one batch."""
    return batch_size * seq_len * coarse_dim * dtype_bytes / (1024 * 1024)


def compression_ratio(edge_hidden_size: int, coarse_dim: int, dtype_bytes: int = 2) -> float:
    """Compare dense edge hidden-state size to transmitted coarse-state size."""
    edge_bytes = edge_hidden_size * dtype_bytes
    coarse_bytes = coarse_dim * dtype_bytes
    return edge_bytes / max(coarse_bytes, 1)
