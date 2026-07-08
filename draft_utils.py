"""Shared utilities for GPT-2 draft generation and selective MDLM refinement."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml

from models_mdlm_wrapper import build_edge_mdlm_model


FEATURE_NAMES = [
    "gpt2_confidence",
    "gpt2_entropy",
    "gpt2_margin",
    "mdlm_confidence",
    "mdlm_entropy",
    "mdlm_margin",
    "score_gap",
    "gpt2_mdlm_agreement",
    "refine_ratio",
    "selected_token_uncertainty",
]


class AcceptGateMLP(nn.Module):
    """Small MLP that predicts whether to accept a LoRA-MDLM refinement."""

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
    selected_token_uncertainty: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build accept-gate features and labels for candidate refinements."""
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
    score_gap = rerank_top2.values[..., 0] - rerank_top2.values[..., -1]

    chosen_gpt2_logprob = cand_gpt2.gather(-1, best_rank).squeeze(-1)
    chosen_mdlm_logprob = cand_mdlm.gather(-1, best_rank).squeeze(-1)
    gpt2_pred = gpt2_top.indices
    ratio_feature = torch.full_like(gpt2_top.values, float(refine_ratio if refine_ratio is not None else 0.0))
    uncertainty_feature = selected_token_uncertainty.float() if selected_token_uncertainty is not None else gpt2_entropy

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
    first_weight = state.get("net.0.weight") if isinstance(state, dict) else None
    if first_weight is not None and first_weight.shape[1] == len(FEATURE_NAMES) + 1:
        # Adapt checkpoints that were saved with one extra feature column.
        state = dict(state)
        state["net.0.weight"] = torch.cat([first_weight[:, :8], first_weight[:, 9:]], dim=1)
    gate.load_state_dict(state)
    gate.eval()
    return gate
