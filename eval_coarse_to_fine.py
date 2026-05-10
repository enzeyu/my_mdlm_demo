"""Evaluate device-only, edge-only, and coarse-to-fine denoising quality."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import yaml

from data_real import build_dataloaders, mask_tokens
from metrics import format_table, gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda, write_csv, write_json
from model_coarse_to_fine import build_model_from_config, coarse_comm_mb, compression_ratio, masked_cross_entropy


def load_config(path: str) -> dict:
    """Load evaluation YAML config."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device() -> torch.device:
    """Use CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(config: dict, ckpt_arg: str | None) -> Path:
    """Use explicit checkpoint path or the default checkpoint in save_dir."""
    if ckpt_arg:
        return Path(ckpt_arg)
    return Path(config["save_dir"]) / "checkpoint.pt"


@torch.no_grad()
def evaluate_mode(model, val_loader, config, tokenizer_info, device, mode: str):
    """Evaluate one mode and return loss, accuracy, latency, and systems metrics."""
    model.eval()
    reset_gpu_memory(device)
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    total_latency = 0.0
    batches = 0

    for batch in itertools.islice(val_loader, int(config["eval_steps"])):
        clean = batch.to(device, non_blocking=True)
        noised, target_mask = mask_tokens(clean, tokenizer_info.mask_token_id, tokenizer_info.pad_token_id, float(config["mask_ratio"]))
        timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)

        sync_if_cuda(device)
        start = now()
        outputs = model(noised, timesteps, mode=mode)
        loss, acc = masked_cross_entropy(outputs["logits"], clean, target_mask)
        sync_if_cuda(device)

        latency = now() - start
        total_latency += latency
        total_loss += float(loss.item())
        total_acc += float(acc.item())
        total_tokens += int(target_mask.sum().item())
        batches += 1

    avg_latency = total_latency / max(batches, 1)
    return {
        "mode": mode,
        "validation_loss": total_loss / max(batches, 1),
        "masked_token_accuracy": total_acc / max(batches, 1),
        "denoising_accuracy": total_acc / max(batches, 1),
        "refinement_latency": avg_latency,
        "tokens_per_sec": total_tokens / max(total_latency, 1e-9),
        "gpu_memory_MB": gpu_memory_mb(device),
        "coarse_comm_MB": coarse_comm_mb(
            int(config["batch_size"]),
            int(config["max_length"]),
            int(config["coarse_dim"]),
            dtype_bytes=2 if config.get("precision", "fp32") in {"fp16", "bf16"} else 4,
        )
        if mode == "coarse_to_fine"
        else 0.0,
        "compression_ratio": compression_ratio(int(config["edge_hidden_size"]), int(config["coarse_dim"])),
        "refinement_gain_over_device_only": 0.0,
    }


def main() -> None:
    """Load checkpoint, evaluate three modes, and save result tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    model = build_model_from_config(config, tokenizer_info.vocab_size, tokenizer_info.pad_token_id).to(device)

    ckpt_path = resolve_checkpoint(config, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    rows = []
    for mode in ["device_only", "edge_only", "coarse_to_fine"]:
        rows.append(evaluate_mode(model, val_loader, config, tokenizer_info, device, mode))

    device_acc = next(row["masked_token_accuracy"] for row in rows if row["mode"] == "device_only")
    for row in rows:
        row["refinement_gain_over_device_only"] = row["masked_token_accuracy"] - device_acc

    save_dir = Path(config["save_dir"])
    write_json(save_dir / "eval_metrics.json", rows)
    write_csv(save_dir / "eval_metrics.csv", rows)
    print(format_table(rows, [
        "mode",
        "validation_loss",
        "masked_token_accuracy",
        "refinement_latency",
        "tokens_per_sec",
        "gpu_memory_MB",
        "coarse_comm_MB",
        "compression_ratio",
        "refinement_gain_over_device_only",
    ]))
    print(f"saved_eval={save_dir / 'eval_metrics.csv'}")


if __name__ == "__main__":
    main()
