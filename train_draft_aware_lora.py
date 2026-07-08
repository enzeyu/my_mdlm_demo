"""Train MDLM LoRA adapters with GPT-2 draft-induced block denoising."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path

import torch

from data_real import build_dataloaders
from lora_utils import (
    DEFAULT_LORA_TARGETS,
    build_draft_aware_inputs,
    build_random_mask_lora_inputs,
    freeze_module,
    inject_lora,
    masked_ce_and_accuracy,
    save_lora_adapter,
    trainable_parameter_report,
)
from metrics import gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda
from draft_utils import (
    choose_device,
    gpt2_teacher_forced_logits,
    load_config,
    load_gpt2,
    load_mdlm,
    uncertainty_from_logits,
    validate_model_surfaces,
)


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


@torch.no_grad()
def eval_draft_context(model, gpt2_model, val_loader, config, tokenizer_info, device: torch.device, max_batches: int) -> dict:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    selected_tokens = 0
    total_tokens = 0
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    block_size = int(config.get("block_size", 1))
    refine_ratio = float(config.get("refine_ratio", 0.2))
    mask_ratio = float(config.get("mask_ratio", 0.15))
    mode = str(config.get("lora_training_mode", "draft_aware"))

    for idx, batch in enumerate(val_loader):
        if idx >= max_batches:
            break
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        if mode == "random_mask":
            mdlm_input, selected = build_random_mask_lora_inputs(
                clean,
                valid_mask,
                int(tokenizer_info.mask_token_id),
                mask_ratio,
            )
            timestep_value = mask_ratio
        elif mode == "draft_aware":
            gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
            draft = gpt2_logits.argmax(dim=-1).clamp_max(int(tokenizer_info.mask_token_id) - 1)
            uncertainty = uncertainty_from_logits(gpt2_logits, score_name)
            mdlm_input, selected = build_draft_aware_inputs(
                clean,
                draft,
                uncertainty,
                valid_mask,
                int(tokenizer_info.mask_token_id),
                refine_ratio,
                block_size,
            )
            timestep_value = refine_ratio
        else:
            raise ValueError(f"Unknown lora_training_mode={mode!r}; expected random_mask or draft_aware")
        timesteps = torch.full((clean.size(0),), max(float(timestep_value), 1e-4), device=device)
        logits = model(mdlm_input, timesteps)["logits"].float()
        stats = masked_ce_and_accuracy(logits, clean, selected)
        tokens = int(stats["tokens"])
        if tokens:
            loss_sum += float(stats["loss"].item()) * tokens
            top1_sum += float(stats["token_acc"]) * tokens
            top5_sum += float(stats["top5_acc"]) * tokens
            selected_tokens += tokens
        total_tokens += int(valid_mask.sum().item())

    model.train()
    denom = max(selected_tokens, 1)
    return {
        "draft_context_loss": loss_sum / denom,
        "lora_training_mode": mode,
        "token_acc": top1_sum / denom,
        "top5_acc": top5_sum / denom,
        "selected_token_ratio": selected_tokens / max(total_tokens, 1),
        "eval_tokens": selected_tokens,
    }


def write_summary(path: Path, config: dict, param_report: dict, targets: list[str], train_rows: list[dict], eval_rows: list[dict]) -> None:
    last_train = train_rows[-1] if train_rows else {}
    last_eval = eval_rows[-1] if eval_rows else {}
    lines = [
        "# MDLM LoRA Training Summary",
        "",
        f"- lora_training_mode: `{config.get('lora_training_mode', 'draft_aware')}`",
        f"- Device GPT-2: `{config.get('device_model_name_or_path')}`",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- train_steps: `{int(config.get('train_steps', 10000))}`",
        f"- block_size: `{int(config.get('block_size', 1))}`",
        f"- refine_ratio: `{float(config.get('refine_ratio', 0.2))}`",
        f"- LoRA rank/alpha/dropout: `{config.get('lora_r', 8)}/{config.get('lora_alpha', 16)}/{config.get('lora_dropout', 0.05)}`",
        f"- total_parameters: `{param_report['total_parameters']}`",
        f"- trainable_parameters: `{param_report['trainable_parameters']}`",
        f"- trainable_ratio: `{param_report['trainable_ratio']:.8f}`",
        "",
        "## LoRA Target Modules",
        "",
        *[f"- `{name}`" for name in targets],
        "",
        "## Last Metrics",
        "",
        f"- train_loss: `{last_train.get('loss', '')}`",
        f"- train_token_acc: `{last_train.get('token_acc', '')}`",
        f"- train_top5_acc: `{last_train.get('top5_acc', '')}`",
        f"- eval_draft_context_loss: `{last_eval.get('draft_context_loss', '')}`",
        f"- eval_token_acc: `{last_eval.get('token_acc', '')}`",
        f"- eval_top5_acc: `{last_eval.get('top5_acc', '')}`",
        "",
        "Run `eval_final_refinement.py` for the final GPT-2 + LoRA-MDLM refinement comparison.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(config_path: str, train_steps: int | None = None) -> list[dict]:
    config = load_config(config_path)
    if train_steps is not None:
        config["train_steps"] = int(train_steps)
    config["mdlm_ckpt"] = None
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    train_loader, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, _ = load_gpt2(config, tokenizer, device)
    mdlm_model = load_mdlm(config, tokenizer_info, device, None).to(device)
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))

    freeze_module(gpt2_model)
    freeze_module(mdlm_model)
    target_names = config.get("lora_target_modules", DEFAULT_LORA_TARGETS)
    targets = inject_lora(
        mdlm_model,
        target_names=target_names,
        r=int(config.get("lora_r", 8)),
        alpha=float(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.05)),
    )
    param_report = trainable_parameter_report(mdlm_model)
    print(
        "lora_parameter_report "
        f"total={param_report['total_parameters']} trainable={param_report['trainable_parameters']} "
        f"ratio={param_report['trainable_ratio']:.8f}",
        flush=True,
    )
    print("lora_target_modules=" + ",".join(targets), flush=True)

    mdlm_model.train()
    optimizer = torch.optim.AdamW(
        [param for param in mdlm_model.parameters() if param.requires_grad],
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    log_interval = int(config.get("log_interval", config.get("log_every", 100)))
    eval_interval = int(config.get("eval_interval", config.get("eval_steps", 1000)))
    max_eval_batches = int(config.get("eval_batches", min(20, int(config.get("eval_steps", 1000)))))
    block_size = int(config.get("block_size", 1))
    refine_ratio = float(config.get("refine_ratio", 0.2))
    mask_ratio = float(config.get("mask_ratio", 0.15))
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    mode = str(config.get("lora_training_mode", "draft_aware"))
    if mode not in {"random_mask", "draft_aware"}:
        raise ValueError(f"Unknown lora_training_mode={mode!r}; expected random_mask or draft_aware")
    optimizer.zero_grad(set_to_none=True)

    try:
        for step, batch in enumerate(cycle(train_loader), start=1):
            if step > int(config.get("train_steps", 10000)):
                break
            step_start = now()
            reset_gpu_memory(device)
            clean = batch.to(device, non_blocking=True)
            valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
            with torch.no_grad():
                if mode == "random_mask":
                    mdlm_input, selected = build_random_mask_lora_inputs(
                        clean,
                        valid_mask,
                        int(tokenizer_info.mask_token_id),
                        mask_ratio,
                    )
                    timestep_value = mask_ratio
                else:
                    gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
                    draft = gpt2_logits.argmax(dim=-1).clamp_max(int(tokenizer_info.mask_token_id) - 1)
                    uncertainty = uncertainty_from_logits(gpt2_logits, score_name)
                    mdlm_input, selected = build_draft_aware_inputs(
                        clean,
                        draft,
                        uncertainty,
                        valid_mask,
                        int(tokenizer_info.mask_token_id),
                        refine_ratio,
                        block_size,
                    )
                    timestep_value = refine_ratio

            timesteps = torch.full((clean.size(0),), max(float(timestep_value), 1e-4), device=device)
            logits = mdlm_model(mdlm_input, timesteps)["logits"].float()
            stats = masked_ce_and_accuracy(logits, clean, selected)
            loss = stats["loss"] / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in mdlm_model.parameters() if param.requires_grad],
                    float(config.get("grad_clip", 0.5)),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            sync_if_cuda(device)
            row = {
                "step": step,
                "loss": float(stats["loss"].item()),
                "lora_training_mode": mode,
                "draft_context_loss": float(stats["loss"].item()),
                "token_acc": float(stats["token_acc"]),
                "top5_acc": float(stats["top5_acc"]),
                "selected_token_ratio": int(selected.sum().item()) / max(int(valid_mask.sum().item()), 1),
                "block_size": block_size,
                "refine_ratio": refine_ratio,
                "lr": optimizer.param_groups[0]["lr"],
                "gpu_memory_MB": gpu_memory_mb(device),
                "step_time": now() - step_start,
            }
            train_rows.append(row)
            if log_interval > 0 and (step == 1 or step % log_interval == 0):
                print(
                    " ".join(f"{key}={value}" for key, value in row.items()),
                    flush=True,
                )
            if eval_interval > 0 and (step == 1 or step % eval_interval == 0):
                eval_row = {"step": step, **eval_draft_context(mdlm_model, gpt2_model, val_loader, config, tokenizer_info, device, max_eval_batches)}
                eval_rows.append(eval_row)
                print("eval " + " ".join(f"{key}={value}" for key, value in eval_row.items()), flush=True)
                save_csv(save_dir / "train_metrics.csv", train_rows)
                save_json(save_dir / "train_metrics.json", {"train": train_rows})
                save_csv(save_dir / "eval_metrics.csv", eval_rows)
                save_json(save_dir / "eval_metrics.json", {"eval": eval_rows})
    except torch.cuda.OutOfMemoryError as exc:
        print("CUDA OOM: lower batch_size or max_length in the YAML config and resume.", flush=True)
        raise exc

    metadata = {
        "lora_r": int(config.get("lora_r", 8)),
        "lora_alpha": float(config.get("lora_alpha", 16)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "lora_training_mode": mode,
        "lora_target_modules": targets,
        "param_report": param_report,
        "base_edge_model": config.get("pretrained_edge_path", config.get("edge_model_name_or_path")),
    }
    save_lora_adapter(mdlm_model, save_dir / "lora_adapter", metadata)
    save_csv(save_dir / "train_metrics.csv", train_rows)
    save_json(save_dir / "train_metrics.json", {"train": train_rows, "lora": metadata})
    save_csv(save_dir / "eval_metrics.csv", eval_rows)
    save_json(save_dir / "eval_metrics.json", {"eval": eval_rows})
    save_json(save_dir / "best_config.json", config)
    write_summary(save_dir / "draft_aware_lora_summary.md", config, param_report, targets, train_rows, eval_rows)
    print(f"saved_lora_adapter={save_dir / 'lora_adapter'}", flush=True)
    return train_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.train_steps)


if __name__ == "__main__":
    main()
