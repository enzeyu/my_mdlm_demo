"""Train a real PyTorch coarse-to-fine masked diffusion LM prototype."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from data_real import build_dataloaders, mask_tokens
from metrics import append_jsonl, gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda, write_csv, write_json
from model_coarse_to_fine import (
    build_model_from_config,
    coarse_alignment_loss,
    coarse_comm_mb,
    compression_ratio,
    masked_cross_entropy,
)


def load_config(path: str) -> dict:
    """Load the YAML training configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, batch, config, tokenizer_info, optimizer, device):
    """Run one masked diffusion training step and return scalar diagnostics."""
    clean = batch.to(device, non_blocking=True)
    noised, target_mask = mask_tokens(
        clean,
        tokenizer_info.mask_token_id,
        tokenizer_info.pad_token_id,
        float(config["mask_ratio"]),
    )
    timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)

    outputs = model(noised, timesteps, mode="coarse_to_fine")
    edge_loss, edge_acc = masked_cross_entropy(outputs["logits"], clean, target_mask)
    device_loss, device_acc = masked_cross_entropy(outputs["device_logits"], clean, target_mask)
    align_loss = coarse_alignment_loss(outputs["coarse"], outputs["edge_coarse"], target_mask)
    loss = edge_loss + float(config["lambda_align"]) * align_loss + float(config.get("lambda_device", 0.2)) * device_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "edge_loss": float(edge_loss.detach().item()),
        "device_loss": float(device_loss.detach().item()),
        "align_loss": float(align_loss.detach().item()),
        "token_acc": float(edge_acc.detach().item()),
        "device_token_acc": float(device_acc.detach().item()),
    }


@torch.no_grad()
def quick_eval(model, val_loader, config, tokenizer_info, device, max_steps: int):
    """Evaluate coarse-to-fine validation loss and accuracy for logging/checkpointing."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in itertools.islice(val_loader, max_steps):
        clean = batch.to(device, non_blocking=True)
        noised, target_mask = mask_tokens(clean, tokenizer_info.mask_token_id, tokenizer_info.pad_token_id, float(config["mask_ratio"]))
        timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)
        outputs = model(noised, timesteps, mode="coarse_to_fine")
        loss, acc = masked_cross_entropy(outputs["logits"], clean, target_mask)
        total_loss += float(loss.item())
        total_acc += float(acc.item())
        count += 1
    model.train()
    return {"val_loss": total_loss / max(count, 1), "val_token_acc": total_acc / max(count, 1)}


def save_checkpoint(path: Path, model, optimizer, config, tokenizer_info, step: int) -> None:
    """Save model, optimizer, config, and tokenizer ids for evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "tokenizer_info": tokenizer_info.__dict__,
        },
        path,
    )


def main() -> None:
    """Train the model and write checkpoints plus CSV/JSON metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    print(f"device={device}")

    train_loader, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    model = build_model_from_config(config, tokenizer_info.vocab_size, tokenizer_info.pad_token_id).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.01)))

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = save_dir / "train_metrics.jsonl"
    metrics_jsonl.unlink(missing_ok=True)
    rows = []
    reset_gpu_memory(device)
    model.train()

    train_iter = itertools.cycle(train_loader)
    train_steps = int(config["train_steps"])
    log_interval = int(config.get("log_interval", 20))
    eval_interval = int(config.get("eval_interval", 100))
    ckpt_path = save_dir / "checkpoint.pt"

    for step in range(1, train_steps + 1):
        batch = next(train_iter)
        sync_if_cuda(device)
        start = now()
        metrics = train_step(model, batch, config, tokenizer_info, optimizer, device)
        sync_if_cuda(device)
        step_time = now() - start

        row = {
            "step": step,
            **metrics,
            "step_time": step_time,
            "gpu_memory_MB": gpu_memory_mb(device),
            "coarse_comm_MB_per_batch": coarse_comm_mb(
                int(config["batch_size"]),
                int(config["max_length"]),
                int(config["coarse_dim"]),
                dtype_bytes=2 if config.get("precision", "fp32") in {"fp16", "bf16"} else 4,
            ),
            "compression_ratio": compression_ratio(int(config["edge_hidden_size"]), int(config["coarse_dim"])),
        }

        if step % eval_interval == 0 or step == train_steps:
            row.update(quick_eval(model, val_loader, config, tokenizer_info, device, int(config.get("eval_log_steps", 5))))
            save_checkpoint(ckpt_path, model, optimizer, config, tokenizer_info, step)

        rows.append(row)
        append_jsonl(metrics_jsonl, row)

        if step % log_interval == 0 or step == 1:
            print(
                f"step={step:04d} loss={row['loss']:.4f} edge_loss={row['edge_loss']:.4f} "
                f"device_loss={row['device_loss']:.4f} align={row['align_loss']:.4f} "
                f"acc={row['token_acc']:.4f} step_time={row['step_time']:.3f}s "
                f"gpu_mem={row['gpu_memory_MB']:.1f}MB"
            )

    write_json(save_dir / "train_metrics.json", rows)
    write_csv(save_dir / "train_metrics.csv", rows)
    save_checkpoint(ckpt_path, model, optimizer, config, tokenizer_info, train_steps)
    print(f"saved_checkpoint={ckpt_path}")
    print(f"saved_metrics={save_dir / 'train_metrics.csv'}")
    print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
