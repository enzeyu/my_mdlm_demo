"""Utilities for AR-draft-induced LoRA fine-tuning of MDLM."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "fc",
    "fc1",
    "fc2",
    "c_fc",
    "c_proj",
    "dense",
    "attn_qkv",
    "attn_out",
    "mlp.0",
    "mlp.2",
]


class LoRALinear(nn.Module):
    """Lightweight LoRA wrapper for Linear layers."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return out + update


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_module(module: nn.Module) -> None:
    module.requires_grad_(False)


def inject_lora(
    model: nn.Module,
    target_names: Iterable[str] | None = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> list[str]:
    """Replace matched Linear modules with LoRA wrappers and return target names."""
    targets = [name.lower() for name in (target_names or DEFAULT_LORA_TARGETS)]
    matched: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        leaf = module_name.rsplit(".", 1)[-1].lower()
        full = module_name.lower()
        if not any(full == target or leaf == target or leaf.endswith(target) or f".{target}" in full for target in targets):
            continue
        if any(skip in full for skip in ("lm_head", "output_layer", "fallback_hidden", "sigma_map")):
            continue
        parent, child_name = _get_parent(model, module_name)
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        matched.append(module_name)
    if not matched:
        raise RuntimeError(
            "No MDLM Linear modules matched LoRA targets. "
            f"Tried targets={targets}. Inspect model.named_modules() and set lora_target_modules."
        )
    return matched


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if "lora_A" in key or "lora_B" in key}


def save_lora_adapter(model: nn.Module, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    try:
        from safetensors.torch import save_file

        save_file(state, path / "adapter_model.safetensors")
    except Exception:
        torch.save(state, path / "adapter_model.pt")
    (path / "adapter_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_lora_adapter(model: nn.Module, path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    metadata = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    safetensors_path = path / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(safetensors_path, device=str(device))
    else:
        state = torch.load(path / "adapter_model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if "lora_" in key]
    if bad_missing or unexpected:
        raise RuntimeError(f"LoRA adapter load mismatch: missing_lora={bad_missing}, unexpected={unexpected}")
    return metadata


def trainable_parameter_report(model: nn.Module) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_ratio": float(trainable / max(total, 1)),
    }


def select_uncertain_blocks(
    uncertainty: torch.Tensor,
    valid_mask: torch.Tensor,
    ratio: float,
    block_size: int,
) -> torch.Tensor:
    """Select non-overlapping high-uncertainty blocks per sequence."""
    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    block_size = max(int(block_size), 1)
    ratio = float(ratio)
    if ratio <= 0:
        return selected
    batch, seq_len = valid_mask.shape
    for row in range(batch):
        valid_count = int(valid_mask[row].sum().item())
        if valid_count == 0:
            continue
        candidates: list[tuple[float, int, int]] = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            block_valid = valid_mask[row, start:end]
            if not bool(block_valid.any()):
                continue
            score = float(uncertainty[row, start:end][block_valid].mean().item())
            candidates.append((score, start, end))
        if not candidates:
            continue
        target_tokens = max(1, int(math.ceil(valid_count * ratio)))
        k = min(len(candidates), max(1, int(math.ceil(target_tokens / block_size))))
        for _, start, end in sorted(candidates, reverse=True)[:k]:
            selected[row, start:end] |= valid_mask[row, start:end]
    return selected


def build_draft_aware_inputs(
    clean: torch.Tensor,
    draft: torch.Tensor,
    uncertainty: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    refine_ratio: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = select_uncertain_blocks(uncertainty, valid_mask, refine_ratio, block_size)
    mdlm_input = draft.clone()
    mdlm_input[selected] = int(mask_token_id)
    return mdlm_input, selected


def build_random_mask_lora_inputs(
    clean: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build standard MDLM random-mask denoising inputs for LoRA training."""
    selected = (torch.rand(clean.shape, device=clean.device) < float(mask_ratio)) & valid_mask
    missing = ~selected.any(dim=1)
    if bool(missing.any()):
        for row in torch.where(missing)[0].tolist():
            valid_pos = torch.where(valid_mask[row])[0]
            if valid_pos.numel() > 0:
                pos = valid_pos[torch.randint(valid_pos.numel(), (1,), device=clean.device)]
                selected[row, pos] = True
    mdlm_input = clean.clone()
    mdlm_input[selected] = int(mask_token_id)
    return mdlm_input, selected


def masked_ce_and_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict:
    if not bool(mask.any()):
        zero = logits.new_tensor(0.0)
        return {"loss": zero, "token_acc": 0.0, "top5_acc": 0.0, "tokens": 0}
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    vocab = selected_logits.size(-1)
    loss = F.cross_entropy(selected_logits.view(-1, vocab), selected_labels.view(-1))
    pred = selected_logits.argmax(dim=-1)
    top5 = selected_logits.topk(min(5, vocab), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
    tokens = int(selected_labels.numel())
    return {
        "loss": loss,
        "token_acc": float(pred.eq(selected_labels).float().mean().item()),
        "top5_acc": float(top5.float().mean().item()),
        "tokens": tokens,
    }


def block_exact_match(pred: torch.Tensor, labels: torch.Tensor, selected: torch.Tensor, block_size: int) -> tuple[int, int]:
    good = 0
    total = 0
    batch, seq_len = selected.shape
    block_size = max(int(block_size), 1)
    for row in range(batch):
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            mask = selected[row, start:end]
            if not bool(mask.any()):
                continue
            total += 1
            if bool(pred[row, start:end][mask].eq(labels[row, start:end][mask]).all()):
                good += 1
    return good, total
