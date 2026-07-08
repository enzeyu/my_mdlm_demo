"""Train a utility-aware accept gate for GPT-2 + LoRA-MDLM refinement."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from data_real import build_dataloaders
from lora_utils import DEFAULT_LORA_TARGETS, freeze_module, inject_lora, load_lora_adapter, select_uncertain_blocks
from draft_utils import (
    FEATURE_NAMES,
    build_gate,
    candidate_rerank_features,
    choose_device,
    gpt2_teacher_forced_logits,
    load_config,
    load_gpt2,
    load_mdlm,
    uncertainty_from_logits,
    validate_model_surfaces,
)
from metrics import now, sync_if_cuda


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else ["step"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_lora_mdlm(config: dict, tokenizer_info, device: torch.device, lora_path: str):
    path = Path(lora_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {path}")
    model = load_mdlm(config, tokenizer_info, device, None).to(device)
    freeze_module(model)
    metadata = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    inject_lora(
        model,
        target_names=metadata.get("lora_target_modules", config.get("lora_target_modules", DEFAULT_LORA_TARGETS)),
        r=int(metadata.get("lora_r", config.get("lora_r", 8))),
        alpha=float(metadata.get("lora_alpha", config.get("lora_alpha", 16))),
        dropout=float(metadata.get("lora_dropout", config.get("lora_dropout", 0.05))),
    )
    load_lora_adapter(model, path, device)
    model.eval()
    return model


def train(config_path: str, train_steps: int | None = None) -> list[dict]:
    config = load_config(config_path)
    if train_steps is not None:
        config["gate_train_steps"] = int(train_steps)
    config["mdlm_ckpt"] = None
    lora_path = str(config.get("draft_aware_lora_path", "results/wikitext2_draft_aware_lora/lora_adapter"))
    ratio = float(config.get("refine_ratio", 0.2))
    block_size = int(config.get("block_size", 1))
    top_k = int(config.get("candidate_top_k", 20))
    lambda_gpt2 = float(config.get("lambda_gpt2", 0.5))
    lambda_mdlm = float(config.get("lambda_mdlm", 0.5))
    positive_weight = float(config.get("positive_weight", 1.0))
    negative_weight = float(config.get("negative_weight", 3.0))
    threshold = float(config.get("gate_threshold", config.get("accept_threshold", 0.5)))

    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    train_loader, _, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    mdlm_model = load_lora_mdlm(config, tokenizer_info, device, lora_path)
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    freeze_module(gpt2_model)
    freeze_module(mdlm_model)
    gpt2_model.eval()
    mdlm_model.eval()

    gate = build_gate(config).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(config.get("gate_lr", 1e-4)))
    save_dir = Path(config["save_dir"])
    metrics: list[dict] = []
    total_samples = 0
    total_positive = 0
    total_negative = 0
    total_correct = 0
    total_seen = 0
    start_time = now()
    log_every = int(config.get("log_interval", config.get("log_every", 100)))

    for step, batch in enumerate(cycle(train_loader), start=1):
        if step > int(config.get("gate_train_steps", 10000)):
            break
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        with torch.no_grad():
            gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
            gpt2_pred = gpt2_logits.argmax(dim=-1)
            draft = gpt2_pred.clamp_max(int(tokenizer_info.mask_token_id) - 1)
            uncertainty = uncertainty_from_logits(gpt2_logits, str(config.get("uncertainty_score", "inverse_confidence")))
            refine_mask = select_uncertain_blocks(uncertainty, valid_mask, ratio, block_size)
            masked_draft = draft.clone()
            masked_draft[refine_mask] = int(tokenizer_info.mask_token_id)
            timesteps = torch.full((clean.size(0),), max(ratio, 1e-4), device=device)
            mdlm_logits = mdlm_model(masked_draft, timesteps)["logits"].float()
            feat = candidate_rerank_features(
                gpt2_logits,
                mdlm_logits,
                clean,
                refine_mask,
                top_k,
                lambda_gpt2,
                lambda_mdlm,
                refine_ratio=ratio,
                selected_token_uncertainty=uncertainty,
            )
            trainable_mask = feat["trainable_mask"]

        if bool(trainable_mask.any()):
            features = feat["features"][trainable_mask].detach()
            labels = feat["accept_label"][trainable_mask].detach()
            logits = gate(features)
            sample_weights = torch.where(
                labels.gt(0.5),
                torch.full_like(labels, positive_weight),
                torch.full_like(labels, negative_weight),
            )
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (raw_loss * sample_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            probs = torch.sigmoid(logits.detach())
            pred = probs.gt(threshold).float()
            correct = int(pred.eq(labels).sum().item())
            sample_count = int(labels.numel())
            positive_count = int(labels.sum().item())
            negative_count = sample_count - positive_count
            total_samples += sample_count
            total_positive += positive_count
            total_negative += negative_count
            total_correct += correct
            total_seen += sample_count
            row = {
                "step": step,
                "gate_loss": float(loss.item()),
                "gate_accuracy": correct / max(sample_count, 1),
                "positive_ratio": positive_count / max(sample_count, 1),
                "num_gate_train_samples": sample_count,
                "total_gate_train_samples": total_samples,
                "positive_weight": positive_weight,
                "negative_weight": negative_weight,
                "elapsed_sec": now() - start_time,
            }
        else:
            row = {
                "step": step,
                "gate_loss": "",
                "gate_accuracy": "",
                "positive_ratio": "",
                "num_gate_train_samples": 0,
                "total_gate_train_samples": total_samples,
                "positive_weight": positive_weight,
                "negative_weight": negative_weight,
                "elapsed_sec": now() - start_time,
            }
        metrics.append(row)
        if log_every > 0 and (step == 1 or step % log_every == 0):
            print(
                f"accept_gate_train_step={step}/{int(config.get('gate_train_steps', 10000))} "
                f"samples={row['num_gate_train_samples']} total={total_samples} loss={row['gate_loss']}",
                flush=True,
            )

    sync_if_cuda(device)
    summary = {
        "gate_loss": next((row["gate_loss"] for row in reversed(metrics) if row["gate_loss"] != ""), ""),
        "gate_accuracy": total_correct / max(total_seen, 1),
        "positive_ratio": total_positive / max(total_samples, 1),
        "negative_ratio": total_negative / max(total_samples, 1),
        "num_gate_train_samples": total_samples,
        "feature_names": FEATURE_NAMES,
        "gpt2_source": gpt2_source,
        "draft_aware_lora_path": lora_path,
        "config": config,
    }
    save_csv(save_dir / "accept_gate_train_metrics.csv", metrics)
    save_json(save_dir / "accept_gate_train_metrics.json", {"train": metrics, "summary": summary})
    torch.save({"model_state": gate.state_dict(), "config": config, "feature_names": FEATURE_NAMES, "summary": summary}, save_dir / "learned_gate.pt")
    save_json(
        save_dir / "accept_gate_best_config.json",
        {
            "refine_ratio": ratio,
            "block_size": block_size,
            "candidate_top_k": top_k,
            "lambda_gpt2": lambda_gpt2,
            "lambda_mdlm": lambda_mdlm,
            "gate_threshold": threshold,
            "gate_hidden_dim": int(config.get("gate_hidden_dim", config.get("gate_hidden_size", 64))),
            "gate_lr": float(config.get("gate_lr", 1e-4)),
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
            "num_gate_train_samples": total_samples,
        },
    )
    print(f"saved_accept_gate={save_dir / 'learned_gate.pt'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.train_steps)


if __name__ == "__main__":
    main()
