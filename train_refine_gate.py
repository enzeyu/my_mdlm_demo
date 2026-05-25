"""Train a learned accept gate for GPT-2 draft + MDLM candidate reranking."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from data_real import build_dataloaders
from refine_utils import (
    choose_device,
    expand_refine_window,
    gpt2_teacher_forced_logits,
    load_config,
    load_gpt2,
    load_mdlm,
    resolve_mdlm_checkpoint,
    select_by_uncertainty,
    uncertainty_from_logits,
    validate_model_surfaces,
)
from metrics import now, sync_if_cuda
from refine_gate import FEATURE_NAMES, build_gate, candidate_rerank_features


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


def train(config_path: str, train_steps: int | None = None) -> list[dict]:
    config = load_config(config_path)
    if train_steps is not None:
        config["train_steps"] = int(train_steps)
    config["refine_windows"] = [int(config.get("refine_window", 0))]
    ratio = float(config.get("refine_ratio", config.get("refine_ratios", [0.3])[-1]))
    top_k = int(config.get("candidate_top_k", 20))
    lambda_gpt2 = float(config.get("lambda_gpt2", 0.5))
    lambda_mdlm = float(config.get("lambda_mdlm", 0.5))

    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    train_loader, _, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    mdlm_model = load_mdlm(config, tokenizer_info, device, resolve_mdlm_checkpoint(config.get("mdlm_ckpt")))
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    gpt2_model.requires_grad_(False)
    mdlm_model.requires_grad_(False)
    gpt2_model.eval()
    mdlm_model.eval()

    gate = build_gate(config).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(config.get("gate_lr", 3e-4)))
    save_dir = Path(config["save_dir"])
    metrics: list[dict] = []
    total_samples = 0
    total_positive = 0
    total_correct = 0
    total_seen = 0
    start_time = now()
    log_every = int(config.get("log_every", 10))

    for step, batch in enumerate(cycle(train_loader), start=1):
        if step > int(config.get("train_steps", 10000)):
            break
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))

        with torch.no_grad():
            gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
            gpt2_pred = gpt2_logits.argmax(dim=-1)
            draft = gpt2_pred.clamp_max(int(tokenizer_info.mask_token_id) - 1)
            uncertainty = uncertainty_from_logits(gpt2_logits, str(config.get("uncertainty_score", "inverse_confidence")))
            refine_mask = select_by_uncertainty(uncertainty, valid_mask, ratio)
            context_mask = expand_refine_window(refine_mask, valid_mask, int(config.get("refine_window", 0)))
            masked_draft = draft.clone()
            masked_draft[context_mask] = int(tokenizer_info.mask_token_id)
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
            )
            trainable_mask = feat["trainable_mask"]

        if trainable_mask.any():
            features = feat["features"][trainable_mask].detach()
            labels = feat["accept_label"][trainable_mask].detach()
            logits = gate(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            probs = torch.sigmoid(logits.detach())
            pred = probs.gt(float(config.get("accept_threshold", 0.5))).float()
            correct = int(pred.eq(labels).sum().item())
            sample_count = int(labels.numel())
            positive_count = int(labels.sum().item())
            total_samples += sample_count
            total_positive += positive_count
            total_correct += correct
            total_seen += sample_count
            row = {
                "step": step,
                "gate_loss": float(loss.item()),
                "gate_accuracy": correct / max(sample_count, 1),
                "positive_ratio": positive_count / max(sample_count, 1),
                "num_gate_train_samples": sample_count,
                "total_gate_train_samples": total_samples,
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
                "elapsed_sec": now() - start_time,
            }
        metrics.append(row)
        if log_every > 0 and (step == 1 or step % log_every == 0):
            print(
                f"gate_train_step={step}/{int(config.get('train_steps', 10000))} "
                f"samples={row['num_gate_train_samples']} total={total_samples} loss={row['gate_loss']}",
                flush=True,
            )

    sync_if_cuda(device)
    summary = {
        "gate_loss": next((row["gate_loss"] for row in reversed(metrics) if row["gate_loss"] != ""), ""),
        "gate_accuracy": total_correct / max(total_seen, 1),
        "positive_ratio": total_positive / max(total_samples, 1),
        "num_gate_train_samples": total_samples,
        "feature_names": FEATURE_NAMES,
        "gpt2_source": gpt2_source,
        "config": config,
    }
    save_csv(save_dir / "gate_train_metrics.csv", metrics)
    save_json(save_dir / "gate_train_metrics.json", {"train": metrics, "summary": summary})
    torch.save({"model_state": gate.state_dict(), "config": config, "feature_names": FEATURE_NAMES, "summary": summary}, save_dir / "learned_gate.pt")
    save_json(
        save_dir / "best_gate_config.json",
        {
            "refine_ratio": ratio,
            "refine_window": int(config.get("refine_window", 0)),
            "candidate_top_k": top_k,
            "lambda_gpt2": lambda_gpt2,
            "lambda_mdlm": lambda_mdlm,
            "accept_threshold": float(config.get("accept_threshold", 0.5)),
            "gate_hidden_size": int(config.get("gate_hidden_size", 64)),
            "gate_layers": int(config.get("gate_layers", 2)),
            "gate_lr": float(config.get("gate_lr", 3e-4)),
            "num_gate_train_samples": total_samples,
        },
    )
    print(f"saved_gate={save_dir / 'learned_gate.pt'}")
    print(f"saved_gate_train_metrics={save_dir / 'gate_train_metrics.csv'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.train_steps)


if __name__ == "__main__":
    main()
