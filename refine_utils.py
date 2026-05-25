"""Shared utilities for GPT-2 draft + MDLM learned-gate refinement."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from metrics import sync_if_cuda, now
from models_mdlm_wrapper import build_edge_mdlm_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "edge_model_name_or_path" in config and "pretrained_edge_path" not in config:
        config["pretrained_edge_path"] = config["edge_model_name_or_path"]
    config.setdefault("dataset_cache_dir", "/mnt/data/enzeyu/hf_downloads/datasets")
    config.setdefault("hf_local_files_only", True)
    config.setdefault("uncertainty_score", "inverse_confidence")
    config.setdefault("refine_ratios", [0.2, 0.3])
    config.setdefault("refine_window", 0)
    config.setdefault("mask_ratio", 0.15)
    if config.get("edge_model_name_or_path") and "use_pretrained_edge" not in config:
        config["use_pretrained_edge"] = True
    if config.get("edge_model_name_or_path") and "require_pretrained_edge" not in config:
        config["require_pretrained_edge"] = True
    return config


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_mdlm_checkpoint(path: str | None) -> Path | None:
    return Path(path) if path else None


def check_tokenizer_compatibility(mdlm_tokenizer, gpt2_tokenizer, gpt2_vocab_size: int) -> None:
    mdlm_vocab_size = len(mdlm_tokenizer)
    if gpt2_vocab_size > mdlm_vocab_size:
        raise ValueError(f"GPT-2 vocab_size={gpt2_vocab_size} exceeds MDLM vocab_size={mdlm_vocab_size}")
    if mdlm_tokenizer.eos_token_id != gpt2_tokenizer.eos_token_id:
        raise ValueError(
            f"Tokenizer EOS mismatch: MDLM eos={mdlm_tokenizer.eos_token_id}, "
            f"GPT-2 eos={gpt2_tokenizer.eos_token_id}"
        )
    if mdlm_tokenizer.pad_token_id != gpt2_tokenizer.pad_token_id:
        raise ValueError(
            f"Tokenizer PAD mismatch: MDLM pad={mdlm_tokenizer.pad_token_id}, "
            f"GPT-2 pad={gpt2_tokenizer.pad_token_id}"
        )
    if mdlm_vocab_size != gpt2_vocab_size:
        mask_id = int(mdlm_tokenizer.mask_token_id)
        if mdlm_vocab_size == gpt2_vocab_size + 1 and mask_id == gpt2_vocab_size:
            print(
                "tokenizer_compatibility=ok "
                f"gpt2_vocab={gpt2_vocab_size} mdlm_vocab={mdlm_vocab_size} "
                f"extra_mdlm_mask_id={mask_id}; GPT-2 candidates stay within base vocab"
            )
        else:
            raise ValueError(f"Tokenizer vocab mismatch: GPT-2={gpt2_vocab_size}, MDLM={mdlm_vocab_size}")


def load_gpt2(config: dict, tokenizer, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(config["device_model_name_or_path"])
    local_files_only = bool(config.get("hf_local_files_only", True))
    gpt2_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    check_tokenizer_compatibility(tokenizer, gpt2_tokenizer, model.config.vocab_size)
    model.requires_grad_(False)
    model.eval()
    print(f"Loaded device GPT-2 from pretrained path: {model_path}")
    return model.to(device), model_path


def load_mdlm(config: dict, tokenizer_info, device: torch.device, ckpt_path: Path | None):
    model = build_edge_mdlm_model(config, tokenizer_info.vocab_size, tokenizer_info.pad_token_id, tokenizer_info.mask_token_id).to(device)
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")
    print(f"Loaded edge MDLM from pretrained path: {config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}")
    if ckpt_path is not None and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ("model_state", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"checkpoint_load_warning missing={len(missing)} unexpected={len(unexpected)}")
        print(f"checkpoint_load_status=loaded path={ckpt_path}")
    elif ckpt_path is not None:
        print(f"checkpoint_load_skipped missing_path={ckpt_path}")
    else:
        print("No MDLM checkpoint provided; using pretrained edge model only")
    model.eval()
    return model


def validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device: torch.device, max_length: int) -> None:
    sample = torch.full((1, min(8, max_length)), int(tokenizer_info.mask_token_id), device=device, dtype=torch.long)
    timesteps = torch.ones((1,), device=device)
    with torch.no_grad():
        mdlm_vocab = int(mdlm_model(sample, timesteps)["logits"].shape[-1])
    tokenizer_vocab = int(tokenizer_info.vocab_size)
    gpt2_vocab = int(gpt2_model.config.vocab_size)
    print(
        "eval_surface_check "
        f"tokenizer_vocab={tokenizer_vocab} gpt2_vocab={gpt2_vocab} mdlm_logits_vocab={mdlm_vocab} "
        f"pad_token_id={int(tokenizer_info.pad_token_id)} mask_token_id={int(tokenizer_info.mask_token_id)} "
        f"tokenizer_mask_id={tokenizer.mask_token_id}"
    )
    if mdlm_vocab != tokenizer_vocab:
        raise ValueError(f"MDLM logits vocab={mdlm_vocab} does not match tokenizer vocab={tokenizer_vocab}")


@torch.no_grad()
def gpt2_teacher_forced_logits(gpt2_model, clean: torch.Tensor, eos_token_id: int):
    shifted = clean.new_empty(clean.shape)
    shifted[:, 0] = eos_token_id
    shifted[:, 1:] = clean[:, :-1]
    attention_mask = torch.ones_like(shifted)
    return gpt2_model(input_ids=shifted, attention_mask=attention_mask).logits.float()


def uncertainty_from_logits(logits: torch.Tensor, score_name: str) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    if score_name == "entropy":
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return -(probs * log_probs).sum(dim=-1)
    if score_name == "margin":
        top2 = probs.topk(2, dim=-1).values
        return 1.0 - (top2[..., 0] - top2[..., 1])
    return 1.0 - probs.max(dim=-1).values


def select_by_uncertainty(uncertainty: torch.Tensor, valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    coords = valid_mask.nonzero(as_tuple=False)
    count = int(coords.size(0))
    if count == 0 or ratio <= 0.0:
        return selected
    k = min(count, max(1, int(math.ceil(count * ratio))))
    chosen = uncertainty[valid_mask].topk(k, largest=True).indices
    chosen_coords = coords[chosen]
    selected[chosen_coords[:, 0], chosen_coords[:, 1]] = True
    return selected


def select_random(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    coords = valid_mask.nonzero(as_tuple=False)
    count = int(coords.size(0))
    if count == 0 or ratio <= 0.0:
        return selected
    k = min(count, max(1, int(math.ceil(count * ratio))))
    chosen = torch.randperm(count, device=valid_mask.device)[:k]
    chosen_coords = coords[chosen]
    selected[chosen_coords[:, 0], chosen_coords[:, 1]] = True
    return selected


def expand_refine_window(refine_mask: torch.Tensor, valid_mask: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 0 or not refine_mask.any():
        return refine_mask & valid_mask
    expanded = refine_mask.clone()
    for offset in range(1, window + 1):
        expanded[:, offset:] |= refine_mask[:, :-offset]
        expanded[:, :-offset] |= refine_mask[:, offset:]
    return expanded & valid_mask


def add_logits_metrics(acc: dict, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
    if not mask.any():
        return
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    vocab = selected_logits.size(-1)
    loss = F.cross_entropy(selected_logits.view(-1, vocab), selected_labels.view(-1), reduction="sum")
    top1 = selected_logits.argmax(dim=-1)
    top5 = selected_logits.topk(min(5, vocab), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
    acc["loss_sum"] += float(loss.item())
    acc["top1_sum"] += int(top1.eq(selected_labels).sum().item())
    acc["top5_sum"] += int(top5.sum().item())
    acc["tokens"] += int(selected_labels.numel())


def add_gpt2_only_metrics(acc: dict, gpt2_logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, latency: float) -> None:
    add_logits_metrics(acc, gpt2_logits, labels, valid_mask)
    gpt2_pred = gpt2_logits.argmax(dim=-1)
    acc["gpt2_errors"] += int((~gpt2_pred.eq(labels) & valid_mask).sum().item())
    acc["latency"] += latency
    acc["batches"] += 1


def add_mdlm_only_metrics(acc: dict, mdlm_logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, latency: float) -> None:
    top1_before = acc["top1_sum"]
    top5_before = acc["top5_sum"]
    tokens_before = acc["tokens"]
    add_logits_metrics(acc, mdlm_logits, labels, valid_mask)
    acc["refined_top1_sum"] += acc["top1_sum"] - top1_before
    acc["refined_top5_sum"] += acc["top5_sum"] - top5_before
    acc["refined_tokens"] += acc["tokens"] - tokens_before
    acc["latency"] += latency
    acc["batches"] += 1
