"""Train a real PyTorch coarse-to-fine masked diffusion LM prototype."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    masked_lm_metrics,
)

# 加载yaml训练配置文件
def load_config(path: str) -> dict:
    """Load the YAML training configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

# 返回cuda
def choose_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    端侧小模型先在 masked 输入上生成低维 coarse 语义表示，边缘侧大模型利用该 coarse 表示恢复被 mask 的 token
    训练时同时优化边缘恢复能力、端侧恢复能力以及端边语义对齐能力
'''
def train_step(model, batch, config, tokenizer_info, optimizer, device, do_optimizer_step: bool, grad_accum: int):
    """Run one masked diffusion training step and return scalar diagnostics."""
    # 对clean token做masked diffusion corruption, noised是被mask的输入序列, target_mask记录了哪些位置被masked
    clean = batch.to(device, non_blocking=True)
    noised, target_mask = mask_tokens(
        clean,
        tokenizer_info.mask_token_id,
        tokenizer_info.pad_token_id,
        float(config["mask_ratio"]),
    )
    # 每个样本的时间步骤都相同
    timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)

    # 模型输出
    outputs = model(noised, timesteps, mode="coarse_to_fine")
    # 计算边缘大模型的 masked token 恢复损失和准确率
    edge_metrics = masked_lm_metrics(outputs["logits"], clean, target_mask)
    edge_loss = edge_metrics["loss"]
    edge_acc = edge_metrics["top1_acc"]
    # 计算终端小模型的masked token恢复损失 和准确率
    device_metrics = masked_lm_metrics(outputs["device_logits"], clean, target_mask)
    device_loss = device_metrics["loss"]
    device_acc = device_metrics["top1_acc"]
    # 计算边端模型之间的对齐损失(device coarse representation 应该和 edge 侧语义空间一致)
    align_loss = coarse_alignment_loss(outputs["coarse"], outputs["edge_coarse"], target_mask)
    distill_loss = clean.new_tensor(0.0, dtype=torch.float32)
    if outputs.get("device_logits") is not None and float(config.get("lambda_distill", 0.0)) > 0 and target_mask.any():
        distill_vocab = min(outputs["device_logits"].size(-1), outputs["logits"].size(-1))
        student_logp = torch.log_softmax(outputs["device_logits"][target_mask][..., :distill_vocab], dim=-1)
        teacher_p = torch.softmax(outputs["logits"].detach()[target_mask][..., :distill_vocab], dim=-1)
        distill_loss = torch.nn.functional.kl_div(student_logp, teacher_p, reduction="batchmean")

    # 总损失为四者之和: edge/device masked diffusion denoising, coarse alignment, optional distillation.
    loss = (
        edge_loss
        + float(config["lambda_align"]) * align_loss
        + float(config.get("lambda_device", 0.2)) * device_loss
        + float(config.get("lambda_distill", 0.0)) * distill_loss
    )

    (loss / max(grad_accum, 1)).backward()
    if do_optimizer_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(loss.detach().item()),
        "edge_loss": float(edge_loss.detach().item()),
        "train_perplexity": float(math.exp(min(edge_loss.detach().item(), 50.0))),
        "device_loss": float(device_loss.detach().item()),
        "device_perplexity": float(math.exp(min(device_loss.detach().item(), 50.0))),
        "align_loss": float(align_loss.detach().item()),
        "distill_loss": float(distill_loss.detach().item()),
        "token_acc": float(edge_acc.detach().item()),
        "top1_acc": float(edge_acc.detach().item()),
        "top5_acc": float(edge_metrics["top5_acc"].detach().item()),
        "train_top5_acc": float(edge_metrics["top5_acc"].detach().item()),
        "device_token_acc": float(device_acc.detach().item()),
        "device_top5_acc": float(device_metrics["top5_acc"].detach().item()),
    }

# 返回恢复的损失 和 恢复的准确性
@torch.no_grad()
def quick_eval(model, val_loader, config, tokenizer_info, device, max_steps: int):
    """Evaluate coarse-to-fine validation loss and accuracy for logging/checkpointing."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_tokens = 0
    count = 0
    # 从粗到细的推理
    for batch in itertools.islice(val_loader, max_steps):
        clean = batch.to(device, non_blocking=True)
        noised, target_mask = mask_tokens(clean, tokenizer_info.mask_token_id, tokenizer_info.pad_token_id, float(config["mask_ratio"]))
        timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)
        outputs = model(noised, timesteps, mode="coarse_to_fine")
        # 只关注最后的回复能力
        metrics = masked_lm_metrics(outputs["logits"], clean, target_mask)
        num_tokens = int(metrics["num_tokens"])
        total_loss += float(metrics["loss"].item()) * num_tokens
        total_top1 += float(metrics["top1_acc"].item()) * num_tokens
        total_top5 += float(metrics["top5_acc"].item()) * num_tokens
        total_tokens += num_tokens
        count += 1
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    val_top1 = total_top1 / max(total_tokens, 1)
    val_top5 = total_top5 / max(total_tokens, 1)
    return {
        "val_loss": avg_loss,
        "val_perplexity": float(math.exp(min(avg_loss, 50.0))),
        "val_token_acc": val_top1,
        "val_top1_acc": val_top1,
        "val_top5_acc": val_top5,
    }

# 当前模型参数 + 优化器状态 + 训练配置 + tokenizer信息 + 当前训练步数
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
    model = build_model_from_config(
        config,
        tokenizer_info.vocab_size,
        tokenizer_info.pad_token_id,
        tokenizer_info.mask_token_id,
    ).to(device)
    print(f"model_backend={getattr(model, 'backend_name', config.get('model_backend', 'internal_toy'))}")
    print(f"pretrained_edge_loaded={getattr(model, 'pretrained_loaded', False)}")
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")
    print(f"edge_model_is_toy={getattr(model, 'backend_name', 'internal_toy') == 'internal_toy'}")
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.01)))

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = save_dir / "train_metrics.jsonl"
    metrics_jsonl.unlink(missing_ok=True)
    rows = []
    #reset_gpu_memory(device)
    model.train()

    train_iter = itertools.cycle(train_loader)
    train_steps = int(config["train_steps"])
    log_interval = int(config.get("log_interval", 20))
    eval_interval = int(config.get("eval_interval", 100))
    ckpt_path = save_dir / "checkpoint.pt"

    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, train_steps + 1):
        batch = next(train_iter)
        sync_if_cuda(device)
        start = now()
        do_optimizer_step = step % grad_accum == 0
        metrics = train_step(model, batch, config, tokenizer_info, optimizer, device, do_optimizer_step, grad_accum)
        sync_if_cuda(device)
        step_time = now() - start
        edge_hidden_size = int(getattr(model, "edge_hidden_size", config.get("edge_hidden_size", 384) if config.get("edge_hidden_size") != "auto" else 768))

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
            "gradient_accumulation_steps": grad_accum,
            "model_backend": getattr(model, "backend_name", config.get("model_backend", "internal_toy")),
            "pretrained_edge_loaded": getattr(model, "pretrained_loaded", False),
            "compression_ratio": compression_ratio(edge_hidden_size, int(config["coarse_dim"])),
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
