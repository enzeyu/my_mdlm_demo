"""Model factory for the NumPy coarse-to-fine diffusion LM demo."""

from __future__ import annotations

from coarse_device_model import CoarseDeviceConfig, CoarseDeviceDenoiser
from coarse_space import CoarseSemanticSpace, CoarseSpaceConfig
from fine_edge_model import FineEdgeConfig, FineEdgeRefiner


def build_components(config: dict, tokenizer):
    seq_len = int(config["data"]["seq_len"])
    coarse_dim = int(config["coarse_space"]["coarse_dim"])
    token_dim = int(config["coarse_space"].get("token_dim", config["edge_model"]["hidden_size"]))
    total_vocab = int(getattr(tokenizer, "total_vocab_size", tokenizer.vocab_size))
    vocab_size = int(tokenizer.vocab_size)
    seed = int(config.get("seed", 7))

    coarse_space = CoarseSemanticSpace(
        CoarseSpaceConfig(
            vocab_size=vocab_size,
            total_vocab_size=total_vocab,
            token_dim=token_dim,
            coarse_dim=coarse_dim,
            method=config["coarse_space"].get("compression_method", "linear"),
            segment_size=int(config["coarse_space"].get("segment_size", 2)),
            seed=seed,
        )
    )
    device_model = CoarseDeviceDenoiser(
        CoarseDeviceConfig(
            coarse_dim=coarse_dim,
            hidden_size=int(config["device_model"]["hidden_size"]),
            num_layers=int(config["device_model"]["num_layers"]),
            num_heads=int(config["device_model"].get("num_heads", 4)),
            seq_len=seq_len,
            lr=float(config["training"].get("device_lr", config["training"]["lr"])),
            seed=seed + 11,
        )
    )
    edge_model = FineEdgeRefiner(
        FineEdgeConfig(
            vocab_size=vocab_size,
            total_vocab_size=total_vocab,
            seq_len=seq_len,
            coarse_dim=coarse_dim,
            hidden_size=int(config["edge_model"]["hidden_size"]),
            num_layers=int(config["edge_model"]["num_layers"]),
            num_heads=int(config["edge_model"].get("num_heads", 4)),
            conditioning=config["edge_model"].get("conditioning", "adapter"),
            lr=float(config["training"].get("edge_lr", config["training"]["lr"])),
            seed=seed + 23,
        )
    )
    return coarse_space, device_model, edge_model
