"""Small PyTorch masked diffusion language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    pad_token_id: int


class MaskedDiffusionLM(nn.Module):
    """A compact Transformer denoiser conditioned on a scalar mask probability."""

    def __init__(self, config: DiffusionLMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, noise_prob: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        if noise_prob.ndim == 1:
            noise_prob = noise_prob[:, None]
        hidden = self.token_embed(input_ids) + self.pos_embed(positions)
        hidden = hidden + self.time_mlp(noise_prob.float()).unsqueeze(1)
        padding_mask = input_ids.eq(self.config.pad_token_id)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.lm_head(self.norm(hidden))


def build_model(model_config: dict, vocab_size: int, seq_len: int, pad_token_id: int) -> MaskedDiffusionLM:
    config = DiffusionLMConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config.get("dropout", 0.1)),
        pad_token_id=pad_token_id,
    )
    return MaskedDiffusionLM(config)
