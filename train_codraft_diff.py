"""Train CoDraft-Diff AR-to-MDLM collaboration modes."""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from codraft_utils import (
    build_token_type_ids,
    cap_refine_mask_by_ratio,
    draft_aware_renoise,
    estimate_draft_risk,
    load_codraft_config,
    masked_lm_stats,
    save_csv,
    save_json,
    teacher_forced_gpt2_draft_with_confidence,
)
from data_real import build_dataloaders, mask_tokens
from draft_utils import choose_device, load_gpt2, load_mdlm, validate_model_surfaces
from lora_utils import freeze_module, trainable_parameter_report
from metrics import gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda


def _make_draft_batch(gpt2_model, clean: torch.Tensor, tokenizer_info) -> dict[str, torch.Tensor]:
    draft = teacher_forced_gpt2_draft_with_confidence(gpt2_model, clean, int(tokenizer_info.pad_token_id))
    draft["draft_ids"] = draft["draft_ids"].clamp(max=int(tokenizer_info.mask_token_id) - 1)
    return draft


def _make_risk_inputs(config: dict, draft: dict[str, torch.Tensor], clean: torch.Tensor, tokenizer_info):
    risk_scores, refine_mask, anchor_mask = estimate_draft_risk(
        draft["draft_ids"],
        draft["token_confidence"],
        draft["token_entropy"],
        draft["token_margin"],
        float(config.get("confidence_threshold", 0.5)),
        float(config.get("entropy_threshold", 3.0)),
        float(config.get("margin_threshold", 0.1)),
        expand_window=int(config.get("expand_window", 1)),
        min_span_len=int(config.get("min_span_len", 1)),
        pad_token_id=int(tokenizer_info.pad_token_id),
    )
    valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
    refine_mask = cap_refine_mask_by_ratio(
        refine_mask,
        risk_scores,
        valid_mask,
        float(config.get("refine_ratio", 0.2)),
    )
    noisy_ids, target_labels, target_mask = draft_aware_renoise(
        draft["draft_ids"],
        clean,
        refine_mask,
        anchor_mask,
        int(tokenizer_info.mask_token_id),
        suspicious_keep_prob=float(config.get("suspicious_keep_prob", 0.0)),
        pad_token_id=int(tokenizer_info.pad_token_id),
    )
    token_type_ids = build_token_type_ids(target_mask, anchor_mask)
    return noisy_ids, target_labels, target_mask, risk_scores, token_type_ids, anchor_mask


def _mdlm_forward(model, noisy_ids, target_mask, risk_scores=None, draft=None, token_type_ids=None, timestep_value=0.2):
    timesteps = torch.full((noisy_ids.size(0),), max(float(timestep_value), 1e-4), device=noisy_ids.device)
    kwargs = {}
    if draft is not None:
        kwargs = {
            "risk_scores": risk_scores,
            "token_confidence": draft["token_confidence"],
            "token_entropy": draft["token_entropy"],
            "token_margin": draft["token_margin"],
            "token_type_ids": token_type_ids,
        }
    return model(noisy_ids, timesteps, **kwargs)["logits"].float()


def train_mdlm_mode(config: dict, mode: str) -> list[dict]:
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    train_loader, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, _ = load_gpt2(config, tokenizer, device)
    if mode == "train_draft_refine_adapter":
        config["use_draft_adapter"] = True
        config["use_draft_conditioning"] = True
    else:
        config["use_draft_adapter"] = False
        config["use_draft_conditioning"] = False
        config["freeze_mdlm_backbone"] = False
    mdlm_model = load_mdlm(config, tokenizer_info, device, None).to(device)
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    freeze_module(gpt2_model)
    if bool(config.get("freeze_mdlm_backbone", False)):
        mdlm_model.edge_model.requires_grad_(False)
    mdlm_model.train()

    params = [param for param in mdlm_model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable MDLM parameters. Disable freeze_mdlm_backbone or enable draft adapter.")
    lr = float(config.get("adapter_lr", config.get("lr", 1e-4))) if mode == "train_draft_refine_adapter" else float(config.get("lr", 3e-4))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(config.get("weight_decay", 0.01)))
    save_dir = Path(config["save_dir"]) / mode.replace("train_", "")
    rows: list[dict] = []
    eval_rows: list[dict] = []
    log_every = int(config.get("log_every", config.get("log_interval", 100)))
    eval_every = int(config.get("eval_every", config.get("eval_interval", 1000)))
    train_steps = int(config.get("train_steps", 1000))

    for step, batch in enumerate(cycle(train_loader), start=1):
        if step > train_steps:
            break
        step_start = now()
        reset_gpu_memory(device)
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        draft = None
        risk_scores = None
        token_type_ids = None
        if mode == "train_mdlm_random_mask":
            noisy_ids, target_mask = mask_tokens(
                clean,
                int(tokenizer_info.mask_token_id),
                int(tokenizer_info.pad_token_id),
                float(config.get("mask_ratio", 0.15)),
            )
            target_labels = clean
            timestep_value = float(config.get("mask_ratio", 0.15))
        elif mode == "train_mdlm_direct_draft_context":
            with torch.no_grad():
                draft = _make_draft_batch(gpt2_model, clean, tokenizer_info)
            noisy_ids = draft["draft_ids"]
            target_labels = clean
            target_mask = valid_mask
            timestep_value = float(config.get("refine_ratio", 0.2))
        elif mode == "train_draft_refine_adapter":
            with torch.no_grad():
                draft = _make_draft_batch(gpt2_model, clean, tokenizer_info)
                noisy_ids, target_labels, target_mask, risk_scores, token_type_ids, _ = _make_risk_inputs(
                    config,
                    draft,
                    clean,
                    tokenizer_info,
                )
            timestep_value = float(target_mask.float().mean().item())
        else:
            raise ValueError(f"Unsupported MDLM train mode: {mode}")

        logits = _mdlm_forward(mdlm_model, noisy_ids, target_mask, risk_scores, draft, token_type_ids, timestep_value)
        stats = masked_lm_stats(logits, target_labels, target_mask)
        loss = stats["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(config.get("grad_clip", 1.0)))
        optimizer.step()
        sync_if_cuda(device)
        row = {
            "step": step,
            "mode": mode,
            "loss": float(loss.item()),
            "token_acc": float(stats["top1"]),
            "top5_acc": float(stats["top5"]),
            "selected_token_ratio": float(target_mask.sum().item() / max(int(valid_mask.sum().item()), 1)),
            "lr": lr,
            "gpu_memory_MB": gpu_memory_mb(device),
            "step_time": now() - step_start,
        }
        rows.append(row)
        if log_every > 0 and (step == 1 or step % log_every == 0):
            print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)
        if eval_every > 0 and (step == 1 or step % eval_every == 0):
            eval_row = evaluate_training_loss(mdlm_model, gpt2_model, val_loader, config, tokenizer_info, device, mode)
            eval_row["step"] = step
            eval_rows.append(eval_row)
            print("eval " + " ".join(f"{key}={value}" for key, value in eval_row.items()), flush=True)

    param_report = trainable_parameter_report(mdlm_model)
    payload = {
        "model_state": mdlm_model.state_dict(),
        "config": config,
        "mode": mode,
        "param_report": param_report,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_dir / "checkpoint.pt")
    save_csv(save_dir / "train_metrics.csv", rows)
    save_json(save_dir / "train_metrics.json", {"train": rows, "eval": eval_rows, "param_report": param_report})
    save_json(save_dir / "best_config.json", config)
    print(f"saved_checkpoint={save_dir / 'checkpoint.pt'}", flush=True)
    return rows


@torch.no_grad()
def evaluate_training_loss(mdlm_model, gpt2_model, val_loader, config, tokenizer_info, device, mode: str) -> dict:
    mdlm_model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    token_sum = 0
    selected_sum = 0
    valid_sum = 0
    for idx, batch in enumerate(val_loader):
        if idx >= int(config.get("eval_batches", 10)):
            break
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        draft = None
        risk_scores = None
        token_type_ids = None
        if mode == "train_mdlm_random_mask":
            noisy_ids, target_mask = mask_tokens(clean, int(tokenizer_info.mask_token_id), int(tokenizer_info.pad_token_id), float(config.get("mask_ratio", 0.15)))
            target_labels = clean
            timestep_value = float(config.get("mask_ratio", 0.15))
        elif mode == "train_mdlm_direct_draft_context":
            draft = _make_draft_batch(gpt2_model, clean, tokenizer_info)
            noisy_ids = draft["draft_ids"]
            target_labels = clean
            target_mask = valid_mask
            timestep_value = float(config.get("refine_ratio", 0.2))
        else:
            draft = _make_draft_batch(gpt2_model, clean, tokenizer_info)
            noisy_ids, target_labels, target_mask, risk_scores, token_type_ids, _ = _make_risk_inputs(config, draft, clean, tokenizer_info)
            timestep_value = float(target_mask.float().mean().item())
        logits = _mdlm_forward(mdlm_model, noisy_ids, target_mask, risk_scores, draft, token_type_ids, timestep_value)
        stats = masked_lm_stats(logits, target_labels, target_mask)
        tokens = int(stats["tokens"])
        loss_sum += float(stats["loss"].item()) * tokens
        top1_sum += float(stats["top1"]) * tokens
        top5_sum += float(stats["top5"]) * tokens
        token_sum += tokens
        selected_sum += int(target_mask.sum().item())
        valid_sum += int(valid_mask.sum().item())
    mdlm_model.train()
    return {
        "mode": mode,
        "loss": loss_sum / max(token_sum, 1),
        "token_acc": top1_sum / max(token_sum, 1),
        "top5_acc": top5_sum / max(token_sum, 1),
        "selected_token_ratio": selected_sum / max(valid_sum, 1),
    }


def train_gpt2_ar(config: dict, joint: bool = False) -> list[dict]:
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    train_loader, _, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, _ = load_gpt2(config, tokenizer, device)
    gpt2_model.requires_grad_(True)
    gpt2_model.train()
    optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=float(config.get("device_lr", config.get("lr", 3e-5))))
    rows: list[dict] = []
    save_dir = Path(config["save_dir"]) / ("codraft_joint" if joint else "gpt2_ar")
    for step, batch in enumerate(cycle(train_loader), start=1):
        if step > int(config.get("train_steps", 1000)):
            break
        clean = batch.to(device, non_blocking=True)
        outputs = gpt2_model(input_ids=clean, attention_mask=clean.ne(int(tokenizer_info.pad_token_id)), labels=clean)
        loss = outputs.loss
        if joint:
            logits = outputs.logits.float()
            probs = torch.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values
            pred = probs.argmax(dim=-1)
            valid = clean.ne(int(tokenizer_info.pad_token_id))
            correct = pred.eq(clean).float()
            calibration = F.binary_cross_entropy(conf[valid].clamp(1e-4, 1 - 1e-4), correct[valid])
            low_conf_ratio = conf[valid].lt(float(config.get("confidence_threshold", 0.5))).float().mean()
            budget = (low_conf_ratio - float(config.get("refine_ratio", 0.2))).pow(2)
            loss = (
                loss
                + float(config.get("confidence_calibration_weight", 0.1)) * calibration
                + float(config.get("refine_ratio_budget_weight", 0.01)) * budget
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt2_model.parameters(), float(config.get("grad_clip", 1.0)))
        optimizer.step()
        row = {"step": step, "mode": "train_codraft_joint" if joint else "train_gpt2_ar", "loss": float(loss.item())}
        rows.append(row)
        if step == 1 or step % int(config.get("log_every", 100)) == 0:
            print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": gpt2_model.state_dict(), "config": config}, save_dir / "checkpoint.pt")
    save_csv(save_dir / "train_metrics.csv", rows)
    print(f"saved_checkpoint={save_dir / 'checkpoint.pt'}", flush=True)
    return rows


def run(config_path: str, mode: str, train_steps: int | None = None) -> list[dict]:
    config = load_codraft_config(config_path)
    if train_steps is not None:
        config["train_steps"] = int(train_steps)
    if mode == "train_gpt2_ar":
        return train_gpt2_ar(config, joint=False)
    if mode == "train_codraft_joint":
        return train_gpt2_ar(config, joint=True)
    if mode in {"train_mdlm_random_mask", "train_mdlm_direct_draft_context", "train_draft_refine_adapter"}:
        return train_mdlm_mode(config, mode)
    raise ValueError(f"Unknown mode={mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.mode, args.train_steps)


if __name__ == "__main__":
    main()
