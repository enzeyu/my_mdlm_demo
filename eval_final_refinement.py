"""Evaluate final DART refinement modes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data_real import build_dataloaders
from lora_utils import freeze_module, select_uncertain_blocks
from metrics import format_table, now, sync_if_cuda
from draft_utils import (
    candidate_rerank_features,
    choose_device,
    gpt2_teacher_forced_logits,
    load_gate_checkpoint,
    load_config,
    load_gpt2,
    load_mdlm,
    pad_gpt2_logits,
    uncertainty_from_logits,
    validate_model_surfaces,
)
from train_accept_gate import load_lora_mdlm


COLUMNS = [
    "mode",
    "gate_threshold",
    "top1",
    "top5",
    "ppl",
    "correction_rate",
    "regression_rate",
    "net_correction",
    "accepted_ratio",
    "gate_accuracy",
    "selected_token_ratio",
    "latency",
    "tokens_per_sec",
]


def new_acc() -> dict:
    return {
        "loss_sum": 0.0,
        "top1_sum": 0,
        "top5_sum": 0,
        "tokens": 0,
        "corrected": 0,
        "regressed": 0,
        "gpt2_wrong_selected": 0,
        "gpt2_correct_selected": 0,
        "accepted": 0,
        "refined_tokens": 0,
        "valid_tokens": 0,
        "gate_correct": 0,
        "gate_total": 0,
        "latency": 0.0,
        "batches": 0,
    }


def add_logits(acc: dict, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
    if not bool(mask.any()):
        return
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    vocab = selected_logits.size(-1)
    label_mask = selected_labels.lt(vocab)
    if not bool(label_mask.any()):
        return
    selected_logits = selected_logits[label_mask]
    selected_labels = selected_labels[label_mask]
    loss = F.cross_entropy(selected_logits.view(-1, vocab), selected_labels.view(-1), reduction="sum")
    pred = selected_logits.argmax(dim=-1)
    top5 = selected_logits.topk(min(5, vocab), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
    acc["loss_sum"] += float(loss.item())
    acc["top1_sum"] += int(pred.eq(selected_labels).sum().item())
    acc["top5_sum"] += int(top5.sum().item())
    acc["tokens"] += int(selected_labels.numel())


def add_final(
    acc: dict,
    final_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    refine_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    gpt2_pred: torch.Tensor,
    gate_decision: torch.Tensor | None,
    gate_label: torch.Tensor,
    gate_eval_mask: torch.Tensor,
) -> None:
    add_logits(acc, final_logits, labels, valid_mask)
    final_pred = final_logits.argmax(dim=-1)
    gpt2_correct = gpt2_pred.eq(labels) & refine_mask
    gpt2_wrong = ~gpt2_pred.eq(labels) & refine_mask
    final_correct = final_pred.eq(labels)
    acc["corrected"] += int((gpt2_wrong & final_correct).sum().item())
    acc["regressed"] += int((gpt2_correct & ~final_correct).sum().item())
    acc["gpt2_wrong_selected"] += int(gpt2_wrong.sum().item())
    acc["gpt2_correct_selected"] += int(gpt2_correct.sum().item())
    acc["accepted"] += int((accepted_mask & valid_mask).sum().item())
    acc["refined_tokens"] += int((refine_mask & valid_mask).sum().item())
    acc["valid_tokens"] += int(valid_mask.sum().item())
    if gate_decision is not None and bool(gate_eval_mask.any()):
        acc["gate_correct"] += int(gate_decision[gate_eval_mask].float().eq(gate_label[gate_eval_mask]).sum().item())
        acc["gate_total"] += int(gate_eval_mask.sum().item())


def finalize(mode: str, threshold: float | str, acc: dict) -> dict:
    tokens = max(int(acc["tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    correction = acc["corrected"] / max(acc["gpt2_wrong_selected"], 1)
    regression = acc["regressed"] / max(acc["gpt2_correct_selected"], 1)
    return {
        "mode": mode,
        "gate_threshold": threshold,
        "top1": acc["top1_sum"] / tokens,
        "top5": acc["top5_sum"] / tokens,
        "ppl": float(math.exp(min(loss, 50.0))),
        "correction_rate": correction,
        "regression_rate": regression,
        "net_correction": correction - regression,
        "accepted_ratio": acc["accepted"] / max(acc["refined_tokens"], 1),
        "gate_accuracy": acc["gate_correct"] / max(acc["gate_total"], 1) if acc["gate_total"] else "",
        "selected_token_ratio": acc["refined_tokens"] / max(acc["valid_tokens"], 1),
        "latency": acc["latency"] / max(acc["batches"], 1),
        "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
    }


def free_model(model, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def compute_refine(
    mdlm_model,
    gate,
    mode: str,
    threshold: float,
    draft: torch.Tensor,
    gpt2_logits: torch.Tensor,
    labels: torch.Tensor,
    refine_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    tokenizer_info,
    device: torch.device,
    config: dict,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    ratio = float(config.get("refine_ratio", 0.2))
    masked_draft = draft.clone()
    masked_draft[refine_mask] = int(tokenizer_info.mask_token_id)
    timesteps = torch.full((draft.size(0),), max(ratio, 1e-4), device=device)
    sync_if_cuda(device)
    start = now()
    mdlm_logits = mdlm_model(masked_draft, timesteps)["logits"].float()
    sync_if_cuda(device)
    latency = now() - start
    feat = candidate_rerank_features(
        gpt2_logits,
        mdlm_logits,
        labels,
        refine_mask,
        int(config.get("candidate_top_k", 20)),
        float(config.get("lambda_gpt2", 0.5)),
        float(config.get("lambda_mdlm", 0.5)),
        refine_ratio=ratio,
        selected_token_uncertainty=uncertainty,
    )
    gpt2_padded = pad_gpt2_logits(gpt2_logits, mdlm_logits.size(-1))
    final_logits = gpt2_padded.clone()
    if mode.endswith("_with_rule_gate"):
        gpt2_probs = torch.softmax(gpt2_padded.float(), dim=-1)
        mdlm_probs = torch.softmax(mdlm_logits.float(), dim=-1)
        gpt2_conf = gpt2_probs.max(dim=-1).values
        mdlm_conf = mdlm_probs.max(dim=-1).values
        accepted_mask = refine_mask.clone()
        accepted_mask &= gpt2_conf.lt(float(config.get("tau_gpt2", 0.5)))
        accepted_mask &= mdlm_conf.gt(float(config.get("tau_mdlm", 0.4)))
        accepted_mask &= (mdlm_conf - gpt2_conf).gt(float(config.get("tau_margin", 0.05)))
        gate_decision = accepted_mask
    elif mode.endswith("_with_learned_gate"):
        if gate is None:
            raise ValueError("learned gate evaluation requires a gate checkpoint")
        accept_prob = torch.sigmoid(gate(feat["features"]))
        accepted_mask = refine_mask & accept_prob.gt(float(threshold))
        gate_decision = accepted_mask
    else:
        accepted_mask = refine_mask.clone()
        gate_decision = None

    rows = accepted_mask.nonzero(as_tuple=False)
    if rows.numel() > 0:
        chosen = feat["rerank_token"][accepted_mask]
        current = final_logits[rows[:, 0], rows[:, 1]]
        final_logits[rows[:, 0], rows[:, 1], chosen] = current.max(dim=-1).values + 1.0
    return final_logits, latency, accepted_mask, gate_decision, feat["accept_label"], feat["trainable_mask"]


@torch.no_grad()
def evaluate(config: dict, gate_ckpt: str | None) -> list[dict]:
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    freeze_module(gpt2_model)
    random_lora_path = Path(str(config.get("random_mask_lora_path", "results/wikitext2_random_mask_lora/lora_adapter")))
    gate = load_gate_checkpoint(gate_ckpt, config, device) if gate_ckpt else None
    thresholds = [float(value) for value in config.get("gate_thresholds", [config.get("gate_threshold", config.get("accept_threshold", 0.5))])]

    accs: dict[tuple[str, float | str], dict] = {
        ("gpt2_only", ""): new_acc(),
        ("pretrained_mdlm_refine", ""): new_acc(),
        ("draft_aware_lora_refine_no_gate", ""): new_acc(),
    }
    if random_lora_path.exists():
        accs[("random_mask_lora_refine", "")] = new_acc()
    if gate is not None:
        for threshold in thresholds:
            accs[("draft_aware_lora_refine_with_learned_gate", threshold)] = new_acc()

    eval_steps = int(config.get("eval_steps", 1000))
    ratio = float(config.get("refine_ratio", 0.2))
    block_size = int(config.get("block_size", 1))
    log_every = int(config.get("log_interval", config.get("log_every", 100)))
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))

    def make_gpt2_inputs(batch: torch.Tensor):
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        sync_if_cuda(device)
        gpt2_start = now()
        gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
        sync_if_cuda(device)
        gpt2_latency = now() - gpt2_start
        gpt2_pred = gpt2_logits.argmax(dim=-1)
        draft = gpt2_pred.clamp_max(int(tokenizer_info.mask_token_id) - 1)
        uncertainty = uncertainty_from_logits(gpt2_logits, score_name)
        refine_mask = select_uncertain_blocks(uncertainty, valid_mask, ratio, block_size)
        return clean, valid_mask, gpt2_logits, gpt2_latency, gpt2_pred, draft, uncertainty, refine_mask

    def log_step(pass_name: str, step: int) -> None:
        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0):
            print(f"accept_gate_eval_pass={pass_name} step={step + 1}/{eval_steps}", flush=True)

    for step, batch in enumerate(val_loader):
        if step >= eval_steps:
            break
        log_step("gpt2_only", step)
        clean, valid_mask, gpt2_logits, gpt2_latency, _, _, _, _ = make_gpt2_inputs(batch)
        acc = accs[("gpt2_only", "")]
        add_logits(acc, gpt2_logits, clean, valid_mask)
        acc["valid_tokens"] += int(valid_mask.sum().item())
        acc["latency"] += gpt2_latency
        acc["batches"] += 1

    def eval_mdlm_pass(pass_name: str, model, jobs: list[tuple[str, float | str]]) -> None:
        for step, batch in enumerate(val_loader):
            if step >= eval_steps:
                break
            log_step(pass_name, step)
            clean, valid_mask, gpt2_logits, gpt2_latency, gpt2_pred, draft, uncertainty, refine_mask = make_gpt2_inputs(batch)
            for mode, threshold in jobs:
                final_logits, latency, accepted_mask, gate_decision, gate_label, gate_mask = compute_refine(
                    model,
                    gate,
                    mode,
                    float(threshold) if threshold != "" else float(config.get("gate_threshold", config.get("accept_threshold", 0.5))),
                    draft,
                    gpt2_logits,
                    clean,
                    refine_mask,
                    uncertainty,
                    tokenizer_info,
                    device,
                    config,
                )
                acc = accs[(mode, threshold)]
                add_final(acc, final_logits, clean, valid_mask, refine_mask, accepted_mask, gpt2_pred, gate_decision, gate_label, gate_mask)
                acc["latency"] += gpt2_latency + latency
                acc["batches"] += 1

    pretrained = load_mdlm(config, tokenizer_info, device, None).to(device)
    freeze_module(pretrained)
    pretrained.eval()
    eval_mdlm_pass("pretrained_mdlm", pretrained, [("pretrained_mdlm_refine", "")])
    free_model(pretrained, device)

    if random_lora_path.exists():
        random_lora = load_lora_mdlm(config, tokenizer_info, device, str(random_lora_path))
        eval_mdlm_pass("random_mask_lora", random_lora, [("random_mask_lora_refine", "")])
        free_model(random_lora, device)

    draft_lora = load_lora_mdlm(
        config,
        tokenizer_info,
        device,
        str(config.get("draft_aware_lora_path", "results/wikitext2_draft_aware_lora/lora_adapter")),
    )
    validate_model_surfaces(draft_lora, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    draft_jobs: list[tuple[str, float | str]] = [
        ("draft_aware_lora_refine_no_gate", ""),
    ]
    if gate is not None:
        draft_jobs.extend(("draft_aware_lora_refine_with_learned_gate", threshold) for threshold in thresholds)
    eval_mdlm_pass("draft_aware_lora", draft_lora, draft_jobs)
    free_model(draft_lora, device)

    print(f"gpt2_source={gpt2_source}", flush=True)
    rows = [finalize(mode, threshold, acc) for (mode, threshold), acc in accs.items()]
    rows.sort(key=lambda row: (row["mode"], str(row["gate_threshold"])))
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict], best: dict | None) -> None:
    by_mode = {row["mode"]: row for row in rows if row["mode"] != "draft_aware_lora_refine_with_learned_gate"}
    gpt2 = by_mode.get("gpt2_only")
    no_gate = by_mode.get("draft_aware_lora_refine_no_gate")
    learned = best
    lines = [
        "# DART Final Evaluation",
        "",
        "| Mode | Threshold | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted | Gate Acc | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gate_acc = row["gate_accuracy"] if row["gate_accuracy"] != "" else 0.0
        threshold = row["gate_threshold"] if row["gate_threshold"] != "" else 0.0
        lines.append(
            f"| {row['mode']} | {float(threshold):.2f} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['ppl']:.4f} | {row['correction_rate']:.4f} | {row['regression_rate']:.4f} | "
            f"{row['net_correction']:.4f} | {row['accepted_ratio']:.4f} | {gate_acc:.4f} | "
            f"{row['selected_token_ratio']:.4f} |"
        )
    lines.extend(["", "## Answers", ""])
    if learned is None:
        lines.extend(
            [
                "1. learned gate 是否降低 regression？未评估，缺少 learned gate checkpoint。",
                "2. learned gate 是否让 net_correction 接近 0 或转正？未评估。",
                "3. GPT-2 + draft-aware LoRA-MDLM + gate 是否超过 GPT-2-only？未评估。",
                "4. 当前最佳 threshold 是多少？未评估。",
            ]
        )
    else:
        lowers_reg = bool(no_gate and learned["regression_rate"] < no_gate["regression_rate"])
        net_ok = learned["net_correction"] >= -0.01
        beats_gpt2 = bool(gpt2 and (learned["top1"], learned["top5"]) > (gpt2["top1"], gpt2["top5"]))
        lines.extend(
            [
                f"1. learned gate 是否降低 regression？{'是' if lowers_reg else '否或证据不足'}。",
                f"2. learned gate 是否让 net_correction 接近 0 或转正？{'是' if net_ok else '否'}；net={learned['net_correction']:.6f}。",
                f"3. GPT-2 + draft-aware LoRA-MDLM + gate 是否超过 GPT-2-only？{'是' if beats_gpt2 else '否或证据不足'}。",
                f"4. 当前最佳 threshold 是多少？`{float(learned['gate_threshold']):.2f}`。",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_best_learned(rows: list[dict]) -> dict | None:
    learned = [row for row in rows if row["mode"] == "draft_aware_lora_refine_with_learned_gate"]
    if not learned:
        return None
    return max(learned, key=lambda row: (row["top1"], row["net_correction"], -row["regression_rate"], row["top5"]))


def run(config_path: str, gate_ckpt: str | None, eval_steps: int | None = None, save_dir: str | None = None) -> list[dict]:
    config = load_config(config_path)
    config["mdlm_ckpt"] = None
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)
    if save_dir is not None:
        config["save_dir"] = save_dir
    if gate_ckpt is None and bool(config.get("use_accept_gate", True)):
        candidate = Path(str(config.get("accept_gate_path", "results/accept_gate/learned_gate.pt")))
        if not candidate.exists():
            candidate = Path(config["save_dir"]) / "learned_gate.pt"
        gate_ckpt = str(candidate) if candidate.exists() else None
    rows = evaluate(config, gate_ckpt)
    best = choose_best_learned(rows)
    out_dir = Path(config["save_dir"])
    save_csv(out_dir / "final_eval.csv", rows)
    save_json(out_dir / "final_eval.json", {"benchmark": rows})
    if best is not None:
        save_json(out_dir / "accept_gate_best_config.json", {**best, "gate_ckpt": gate_ckpt})
    write_summary(out_dir / "final_summary.md", rows, best)
    print(format_table(rows, COLUMNS))
    print(f"saved_final_eval={out_dir / 'final_eval.csv'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate_ckpt", default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    run(args.config, args.gate_ckpt, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
