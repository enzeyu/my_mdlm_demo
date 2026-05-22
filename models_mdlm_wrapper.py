"""MDLM-style backend for edge-device masked diffusion LM experiments.

This module keeps the project self-contained while supporting three edge cases:

1. load an MDLM checkpoint from a local path;
2. load a Kuleshov Group MDLM checkpoint through the Hugging Face API, preferring
   `HF_ENDPOINT=https://hf-mirror.com` for mainland China networks;
3. optionally fall back to a randomly initialized MDLM-style DiT denoiser.

The fallback is not the old toy Transformer: it uses an MDLM-style absorbing
mask corruption interface, timestep conditioning, DiT adaptive layer norm
blocks, tied token embeddings, and masked-token SUBS-style loss usage in the
training script.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_MODULES_CACHE = "/tmp/hf_modules_cache"


def prefer_hf_mirror() -> None:
    """Prefer hf-mirror and a writable module cache unless explicitly configured."""
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE)


@dataclass
class MDLMBackendConfig:
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
    mask_token_id: int
    pretrained_edge_path: str | None = None
    use_pretrained_edge: bool = False
    require_pretrained_edge: bool = False
    hf_local_files_only: bool = False
    conditioning: str = "hidden"


class TimestepEmbedder(nn.Module):
    """Small continuous timestep MLP used by MDLM-style denoisers."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.net(timesteps[:, None].float())


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm conditioning on diffusion time."""

    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(time_emb).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        y, _ = self.attn(y, y, y, key_padding_mask=padding_mask, need_weights=False)
        x = x + self.dropout(gate_msa[:, None] * y)
        y = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        y = self.mlp(y)
        return x + self.dropout(gate_mlp[:, None] * y)


class MDLMStyleDenoiser(nn.Module):
    """Compact DiT masked diffusion denoiser compatible with MDLM objectives."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_size: int,
        layers: int,
        heads: int,
        dropout: float,
        pad_token_id: int,
        mask_token_id: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_length, hidden_size)
        self.time_embed = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, heads, dropout) for _ in range(layers)])
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        time_emb = self.time_embed(timesteps)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        if conditioning is not None:
            x = x + conditioning
        padding_mask = input_ids.eq(self.pad_token_id)
        for block in self.blocks:
            x = block(x, time_emb, padding_mask)
        shift, scale = self.final_ada(time_emb).chunk(2, dim=-1)
        hidden = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        logits = self.lm_head(hidden)
        if self.mask_token_id < logits.size(-1):
            logits[..., self.mask_token_id] = -1e4
        return logits, hidden


class HFMDLMEdgeAdapter(nn.Module):
    """Wrapper for official MDLM Hugging Face checkpoints."""

    def __init__(self, model: nn.Module, hidden_size: int, vocab_size: int, mask_token_id: int):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.fallback_hidden = nn.Linear(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "timesteps": timesteps,
            "output_hidden_states": True,
            "return_dict": True,
        }
        try:
            outputs = self.model(**kwargs)
        except TypeError:
            outputs = self.model(input_ids=input_ids, timesteps=timesteps)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if conditioning is not None:
            # Official HF forward signatures do not expose a stable hidden-state
            # injection point, so coarse-to-fine conditioning is applied as a
            # learned residual in the wrapper container after this call.
            pass
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            hidden = hidden_states[-1]
        else:
            hidden = self.fallback_hidden(logits.detach())
        if self.mask_token_id < logits.size(-1):
            logits[..., self.mask_token_id] = -1e4
        return logits, hidden


def _config_get(config: Any, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    if default is None:
        raise AttributeError(f"None of {names} found in pretrained config")
    return int(default)


def _load_hf_mdlm_model(model_name: str, local_files_only: bool = False) -> tuple[nn.Module, Any]:
    """Load an HF MDLM checkpoint with compatibility across Transformers versions."""
    from transformers import AutoConfig, AutoModelForMaskedLM

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    common_kwargs = {
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name, dtype="auto", **common_kwargs)
    except TypeError:
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype="auto", **common_kwargs)
        except TypeError:
            model = AutoModelForMaskedLM.from_pretrained(model_name, **common_kwargs)
    return model, hf_config


def _safe_apply_rotary_pos_emb(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch replacement for MDLM remote code's flash-attn rotary kernel.

    The local `mdlm-no_flashattn` checkpoint still routes rotary embeddings
    through `flash_attn.layers.rotary.apply_rotary_emb_qkv_`.  That Triton
    kernel is unnecessary because attention itself is already regular PyTorch,
    and it can surface as delayed CUDA illegal-memory-access failures during
    long fine-tuning runs.  This function preserves the same non-interleaved
    rotary transform for Q/K and leaves V unchanged.
    """
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    rotary_dim = cos.shape[-1] * 2

    qk = qkv[:, :, :2, :, :rotary_dim]
    qk_first, qk_second = qk.chunk(2, dim=-1)
    cos = cos[None, :, None, None, :]
    sin = sin[None, :, None, None, :]
    rotated_qk = torch.cat(
        (
            qk_first * cos - qk_second * sin,
            qk_second * cos + qk_first * sin,
        ),
        dim=-1,
    )
    if rotary_dim < qkv.shape[-1]:
        rotated_qk = torch.cat((rotated_qk, qkv[:, :, :2, :, rotary_dim:]), dim=-1)
    return torch.cat((rotated_qk, qkv[:, :, 2:, :, :]), dim=2)


def _patch_mdlm_rotary_kernel(model: nn.Module) -> bool:
    """Patch HF remote MDLM modules to avoid flash-attn rotary kernels."""
    module = sys.modules.get(model.__class__.__module__)
    if module is None or not hasattr(module, "apply_rotary_pos_emb"):
        return False
    if getattr(module.apply_rotary_pos_emb, "__name__", "") == _safe_apply_rotary_pos_emb.__name__:
        return False
    module.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb
    return True


def _copy_extra_rows(new_tensor: torch.Tensor, old_rows: int, source_row: torch.Tensor) -> None:
    """Initialize newly added vocabulary rows from an existing stable token row."""
    if new_tensor.size(0) <= old_rows:
        return
    with torch.no_grad():
        target = new_tensor[old_rows:]
        source = source_row.reshape((1,) + tuple(target.shape[1:]))
        target.copy_(source.expand_as(target))


def _resize_embedding(embedding: nn.Embedding, new_vocab_size: int, source_token_id: int) -> nn.Embedding:
    """Resize an embedding module while preserving pretrained rows."""
    old_vocab_size, hidden_size = embedding.weight.shape
    if old_vocab_size == new_vocab_size:
        return embedding
    new_embedding = nn.Embedding(
        new_vocab_size,
        hidden_size,
        padding_idx=embedding.padding_idx,
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    rows_to_copy = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_embedding.weight[:rows_to_copy].copy_(embedding.weight[:rows_to_copy])
    source_index = min(source_token_id, old_vocab_size - 1)
    _copy_extra_rows(new_embedding.weight, old_vocab_size, embedding.weight[source_index])
    return new_embedding


def _resize_linear(linear: nn.Linear, new_out_features: int, source_token_id: int) -> nn.Linear:
    """Resize an LM projection while preserving pretrained output rows."""
    old_out_features, in_features = linear.weight.shape
    if old_out_features == new_out_features:
        return linear
    new_linear = nn.Linear(
        in_features,
        new_out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    rows_to_copy = min(old_out_features, new_out_features)
    with torch.no_grad():
        new_linear.weight[:rows_to_copy].copy_(linear.weight[:rows_to_copy])
        if linear.bias is not None and new_linear.bias is not None:
            new_linear.bias[:rows_to_copy].copy_(linear.bias[:rows_to_copy])
    source_index = min(source_token_id, old_out_features - 1)
    _copy_extra_rows(new_linear.weight, old_out_features, linear.weight[source_index])
    if linear.bias is not None and new_linear.bias is not None:
        _copy_extra_rows(new_linear.bias, old_out_features, linear.bias[source_index])
    return new_linear


def _resize_embedding_parameter_layer(module: nn.Module, new_vocab_size: int, source_token_id: int) -> bool:
    """Resize MDLM's custom EmbeddingLayer, which stores weights as `.embedding`."""
    weight = getattr(module, "embedding", None)
    if not isinstance(weight, nn.Parameter):
        return False
    old_vocab_size, hidden_size = weight.shape
    if old_vocab_size == new_vocab_size:
        return True
    new_weight = torch.empty(
        new_vocab_size,
        hidden_size,
        device=weight.device,
        dtype=weight.dtype,
    )
    rows_to_copy = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_weight[:rows_to_copy].copy_(weight[:rows_to_copy])
    source_index = min(source_token_id, old_vocab_size - 1)
    _copy_extra_rows(new_weight, old_vocab_size, weight[source_index])
    module.embedding = nn.Parameter(new_weight)
    return True


def _resize_pretrained_mdlm_vocab(model: nn.Module, vocab_size: int, mask_token_id: int, pad_token_id: int) -> str:
    """Ensure the HF MDLM can consume the added `[MASK]` token id."""
    try:
        input_embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        input_embeddings = None
    source_token_id = pad_token_id if pad_token_id is not None else mask_token_id - 1

    def sync_config_vocab() -> None:
        if hasattr(model, "config"):
            model.config.vocab_size = vocab_size
            model.config.mask_token_id = mask_token_id
            model.config.pad_token_id = pad_token_id
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            if hasattr(backbone, "config"):
                backbone.config.vocab_size = vocab_size
                backbone.config.mask_token_id = mask_token_id
                backbone.config.pad_token_id = pad_token_id
            if hasattr(backbone, "vocab_size"):
                backbone.vocab_size = vocab_size

    if input_embeddings is not None:
        old_vocab = input_embeddings.weight.size(0)
        if old_vocab != vocab_size:
            model.resize_token_embeddings(vocab_size)
            try:
                resized_embeddings = model.get_input_embeddings()
                _copy_extra_rows(
                    resized_embeddings.weight,
                    old_vocab,
                    input_embeddings.weight[min(source_token_id, old_vocab - 1)],
                )
            except (AttributeError, NotImplementedError):
                pass
            sync_config_vocab()
            return f"resized HF token embeddings from {old_vocab} to {vocab_size}"
        sync_config_vocab()
        return f"HF token embeddings already match vocab_size={vocab_size}"

    backbone = getattr(model, "backbone", None)
    vocab_embed = getattr(backbone, "vocab_embed", None)
    output_layer = getattr(backbone, "output_layer", None)
    output_linear = getattr(output_layer, "linear", None)
    if isinstance(vocab_embed, nn.Embedding):
        old_vocab = vocab_embed.weight.size(0)
        if old_vocab != vocab_size:
            backbone.vocab_embed = _resize_embedding(vocab_embed, vocab_size, source_token_id)
            if isinstance(output_linear, nn.Linear):
                output_layer.linear = _resize_linear(output_linear, vocab_size, source_token_id)
            sync_config_vocab()
            return f"resized MDLM vocab_embed from {old_vocab} to {vocab_size}"
        sync_config_vocab()
        return f"MDLM vocab_embed already matches vocab_size={vocab_size}"
    if vocab_embed is not None and isinstance(getattr(vocab_embed, "embedding", None), nn.Parameter):
        old_vocab = vocab_embed.embedding.size(0)
        if old_vocab != vocab_size:
            _resize_embedding_parameter_layer(vocab_embed, vocab_size, source_token_id)
            if isinstance(output_linear, nn.Linear):
                output_layer.linear = _resize_linear(output_linear, vocab_size, source_token_id)
            sync_config_vocab()
            return f"resized MDLM custom vocab_embed from {old_vocab} to {vocab_size}"
        sync_config_vocab()
        return f"MDLM custom vocab_embed already matches vocab_size={vocab_size}"

    return "could not find a resizable HF/MDLM token embedding; assuming checkpoint vocab is compatible"


def _assert_pretrained_mdlm_vocab(model: nn.Module, vocab_size: int, mask_token_id: int) -> None:
    """Fail early if any known MDLM vocab surface cannot represent `[MASK]`."""
    if mask_token_id >= vocab_size:
        raise ValueError(f"mask_token_id={mask_token_id} is outside vocab_size={vocab_size}")

    checks: list[tuple[str, int]] = []
    try:
        input_embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        input_embeddings = None
    if input_embeddings is not None:
        checks.append(("input_embeddings", input_embeddings.weight.size(0)))

    backbone = getattr(model, "backbone", None)
    vocab_embed = getattr(backbone, "vocab_embed", None)
    if isinstance(vocab_embed, nn.Embedding):
        checks.append(("backbone.vocab_embed", vocab_embed.weight.size(0)))
    elif vocab_embed is not None and isinstance(getattr(vocab_embed, "embedding", None), nn.Parameter):
        checks.append(("backbone.vocab_embed.embedding", vocab_embed.embedding.size(0)))

    output_linear = getattr(getattr(backbone, "output_layer", None), "linear", None)
    if isinstance(output_linear, nn.Linear):
        checks.append(("backbone.output_layer.linear", output_linear.out_features))

    bad = [f"{name}={size}" for name, size in checks if size <= mask_token_id or size != vocab_size]
    if bad:
        raise ValueError(
            "Pretrained MDLM vocab mismatch after resize: "
            + ", ".join(bad)
            + f"; expected vocab_size={vocab_size}, mask_token_id={mask_token_id}"
        )


def try_load_pretrained_mdlm(
    path_or_name: str,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    local_files_only: bool = False,
) -> tuple[nn.Module | None, int, str]:
    """Try loading an official MDLM checkpoint from local path or hf-mirror."""
    prefer_hf_mirror()
    try:
        is_local = Path(path_or_name).exists()
        model_name = str(Path(path_or_name)) if is_local else path_or_name
        model, hf_config = _load_hf_mdlm_model(model_name, local_files_only=local_files_only)
        resize_message = _resize_pretrained_mdlm_vocab(model, vocab_size, mask_token_id, pad_token_id)
        _assert_pretrained_mdlm_vocab(model, vocab_size, mask_token_id)
        rotary_message = "patched flash-attn rotary kernel" if _patch_mdlm_rotary_kernel(model) else "rotary kernel already safe"
        hidden_size = _config_get(hf_config, "hidden_size", "hidden_dim", "d_model", "n_embd", default=768)
        message = (
            f"loaded pretrained edge MDLM from {model_name} via HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}; "
            f"{resize_message}; {rotary_message}"
        )
        return HFMDLMEdgeAdapter(model, hidden_size, vocab_size, mask_token_id), hidden_size, message
    except Exception as exc:  # noqa: BLE001 - this is intentionally best effort.
        return None, 0, f"failed to load pretrained edge MDLM from {path_or_name}: {exc}"


class MDLMCoarseToFineModel(nn.Module):
    """Device-lightweight + edge-MDLM wrapper with three evaluation modes."""

    def __init__(self, config: MDLMBackendConfig):
        super().__init__()
        self.config = config
        self.backend_name = "mdlm"
        self.pretrained_loaded = False
        self.load_message = "pretrained loading not requested"

        self.device_model = MDLMStyleDenoiser(
            config.vocab_size,
            config.max_length,
            config.device_hidden_size,
            config.device_layers,
            config.device_heads,
            config.dropout,
            config.pad_token_id,
            config.mask_token_id,
        )
        self.device_to_coarse = nn.Linear(config.device_hidden_size, config.coarse_dim)
        self.device_lm_head = nn.Linear(config.device_hidden_size, config.vocab_size, bias=False)
        self.device_lm_head.weight = self.device_model.token_embed.weight

        edge_model: nn.Module | None = None
        edge_hidden = config.edge_hidden_size
        if config.use_pretrained_edge and config.pretrained_edge_path:
            edge_model, loaded_hidden, message = try_load_pretrained_mdlm(
                config.pretrained_edge_path,
                config.vocab_size,
                config.mask_token_id,
                config.pad_token_id,
                local_files_only=config.hf_local_files_only,
            )
            self.load_message = message
            if edge_model is not None:
                edge_hidden = loaded_hidden
                self.pretrained_loaded = True

        if edge_model is None:
            if config.require_pretrained_edge:
                raise RuntimeError(
                    f"required pretrained edge MDLM was not loaded. {self.load_message}"
                )
            self.load_message = (
                f"{self.load_message}; using randomly initialized MDLM-style DiT edge model"
            )
            edge_model = MDLMStyleDenoiser(
                config.vocab_size,
                config.max_length,
                edge_hidden,
                config.edge_layers,
                config.edge_heads,
                config.dropout,
                config.pad_token_id,
                config.mask_token_id,
            )

        self.edge_hidden_size = edge_hidden
        self.edge_model = edge_model
        self.coarse_to_edge = nn.Linear(config.coarse_dim, edge_hidden)
        self.coarse_to_logits = nn.Linear(edge_hidden, config.vocab_size, bias=False)
        self.edge_to_coarse = nn.Linear(edge_hidden, config.coarse_dim)

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor, mode: str = "coarse_to_fine"):
        if mode == "device_only":
            device_logits, device_hidden = self.device_model(input_ids, timesteps)
            coarse = self.device_to_coarse(device_hidden)
            return {"logits": device_logits, "device_logits": device_logits, "coarse": coarse}

        if mode == "edge_only":
            edge_logits, edge_hidden = self.edge_model(input_ids, timesteps, conditioning=None)
            edge_coarse = self.edge_to_coarse(edge_hidden)
            return {
                "logits": edge_logits,
                "device_logits": None,
                "coarse": None,
                "edge_hidden": edge_hidden,
                "edge_coarse": edge_coarse,
            }

        if mode != "coarse_to_fine":
            raise ValueError(f"Unknown mode: {mode}")

        device_logits, device_hidden = self.device_model(input_ids, timesteps)
        coarse = self.device_to_coarse(device_hidden)
        conditioning = self.coarse_to_edge(coarse)
        if isinstance(self.edge_model, HFMDLMEdgeAdapter):
            edge_logits, edge_hidden = self.edge_model(input_ids, timesteps, conditioning=None)
            cond_logits = self.coarse_to_logits(conditioning)
            common_vocab = min(edge_logits.size(-1), cond_logits.size(-1))
            edge_logits = edge_logits.clone()
            edge_logits[..., :common_vocab] = edge_logits[..., :common_vocab] + cond_logits[..., :common_vocab]
        else:
            edge_logits, edge_hidden = self.edge_model(input_ids, timesteps, conditioning=conditioning)
        edge_coarse = self.edge_to_coarse(edge_hidden)
        return {
            "logits": edge_logits,
            "device_logits": device_logits,
            "coarse": coarse,
            "edge_hidden": edge_hidden,
            "edge_coarse": edge_coarse,
        }


def build_mdlm_backend(config: dict, vocab_size: int, pad_token_id: int, mask_token_id: int) -> MDLMCoarseToFineModel:
    """Build MDLM backend from YAML values and tokenizer metadata."""
    edge_hidden = config.get("edge_hidden_size", 768)
    if edge_hidden == "auto":
        edge_hidden = 768
    edge_layers = config.get("edge_layers", 12)
    if edge_layers == "auto":
        edge_layers = 12
    edge_heads = config.get("edge_heads", 12)
    if edge_heads == "auto":
        edge_heads = 12
    cfg = MDLMBackendConfig(
        vocab_size=vocab_size,
        max_length=int(config["max_length"]),
        coarse_dim=int(config["coarse_dim"]),
        device_hidden_size=int(config["device_hidden_size"]),
        edge_hidden_size=int(edge_hidden),
        device_layers=int(config["device_layers"]),
        edge_layers=int(edge_layers),
        device_heads=int(config.get("device_heads", 4)),
        edge_heads=int(edge_heads),
        dropout=float(config.get("dropout", 0.1)),
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        pretrained_edge_path=config.get("pretrained_edge_path"),
        use_pretrained_edge=bool(config.get("use_pretrained_edge", False)),
        require_pretrained_edge=bool(config.get("require_pretrained_edge", False)),
        hf_local_files_only=bool(config.get("hf_local_files_only", False)),
        conditioning=str(config.get("coarse_conditioning", "hidden")),
    )
    return MDLMCoarseToFineModel(cfg)


def mdlm_subs_parameterization_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    noised_input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """MDLM SUBS-style loss: supervise only absorbing-mask denoising positions."""
    del noised_input_ids, mask_token_id
    if target_mask.sum().item() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[target_mask], labels[target_mask])
