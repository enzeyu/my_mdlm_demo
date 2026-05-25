"""Evaluate candidate reranking with no gate, rule gate, and learned accept gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data_real import build_dataloaders, mask_tokens
from refine_utils import (
    add_gpt2_only_metrics,
    add_mdlm_only_metrics,
    choose_device,
    expand_refine_window,
    gpt2_teacher_forced_logits,
    load_config,
    load_gpt2,
    load_mdlm,
    resolve_mdlm_checkpoint,
    select_by_uncertainty,
    select_random,
    uncertainty_from_logits,
    validate_model_surfaces,
)
from metrics import format_table, now, sync_if_cuda
from refine_gate import candidate_rerank_features, load_gate_checkpoint, pad_gpt2_logits


EVAL_COLUMNS = [
    "mode",
    "refine_ratio",
    "candidate_top_k",
    "top1",
    "top5",
    "ppl",
    "correction_rate",
    "regression_rate",
    "net_correction",
    "accepted_ratio",
    "gate_accuracy",
    "error_detection_precision",
    "error_detection_recall",
    "candidate_coverage",
    "latency",
    "tokens_per_sec",
]


def new_acc() -> dict:
    return {
        "loss_sum": 0.0,
        "top1_sum": 0,
        "top5_sum": 0,
        "tokens": 0,
        "refined_top1_sum": 0,
        "refined_top5_sum": 0,
        "refined_tokens": 0,
        "corrected": 0,
        "regressed": 0,
        "gpt2_wrong_selected": 0,
        "gpt2_correct_selected": 0,
        "selected_errors": 0,
        "gpt2_errors": 0,
        "accepted": 0,
        "candidate_hits": 0,
        "candidate_total": 0,
        "gate_correct": 0,
        "gate_total": 0,
        "latency": 0.0,
        "batches": 0,
    }


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


def add_refined_metrics(
    acc: dict,
    final_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    refine_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    gpt2_pred: torch.Tensor,
    candidate_hit_mask: torch.Tensor,
    gate_decision: torch.Tensor | None,
    gate_label: torch.Tensor,
    gate_eval_mask: torch.Tensor,
) -> None:
    add_logits_metrics(acc, final_logits, labels, valid_mask)
    gpt2_correct_all = gpt2_pred.eq(labels) & valid_mask
    gpt2_wrong_all = ~gpt2_pred.eq(labels) & valid_mask
    acc["gpt2_errors"] += int(gpt2_wrong_all.sum().item())
    acc["accepted"] += int((accepted_mask & valid_mask).sum().item())
    acc["refined_tokens"] += int((refine_mask & valid_mask).sum().item())
    if refine_mask.any():
        final_pred = final_logits.argmax(dim=-1)
        refined_correct = final_pred.eq(labels)
        gpt2_correct = gpt2_correct_all[refine_mask]
        gpt2_wrong = gpt2_wrong_all[refine_mask]
        acc["corrected"] += int((gpt2_wrong & refined_correct[refine_mask]).sum().item())
        acc["regressed"] += int((gpt2_correct & ~refined_correct[refine_mask]).sum().item())
        acc["gpt2_wrong_selected"] += int(gpt2_wrong.sum().item())
        acc["gpt2_correct_selected"] += int(gpt2_correct.sum().item())
        acc["selected_errors"] += int(gpt2_wrong.sum().item())
        acc["candidate_hits"] += int((candidate_hit_mask & refine_mask).sum().item())
        acc["candidate_total"] += int(refine_mask.sum().item())
    if gate_decision is not None and gate_eval_mask.any():
        acc["gate_correct"] += int(gate_decision[gate_eval_mask].float().eq(gate_label[gate_eval_mask]).sum().item())
        acc["gate_total"] += int(gate_eval_mask.sum().item())


@torch.no_grad()
def rerank_refine(
    mdlm_model,
    gate,
    draft: torch.Tensor,
    gpt2_logits: torch.Tensor,
    labels: torch.Tensor,
    refine_mask: torch.Tensor,
    context_mask: torch.Tensor,
    tokenizer_info,
    device: torch.device,
    ratio: float,
    mode: str,
    config: dict,
):
    masked_draft = draft.clone()
    masked_draft[context_mask] = int(tokenizer_info.mask_token_id)
    timesteps = torch.full((draft.size(0),), max(float(ratio), 1e-4), device=device)
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
    )
    gpt2_padded = pad_gpt2_logits(gpt2_logits, mdlm_logits.size(-1))
    final_logits = gpt2_padded.clone()
    rerank_token = feat["rerank_token"]

    if mode in {"candidate_rerank_no_gate", "random_refine"}:
        accepted_mask = refine_mask.clone()
        gate_decision = None
    elif mode == "candidate_rerank_rule_gate":
        gpt2_probs = torch.softmax(gpt2_padded.float(), dim=-1)
        mdlm_probs = torch.softmax(mdlm_logits.float(), dim=-1)
        gpt2_conf = gpt2_probs.max(dim=-1).values
        mdlm_conf = mdlm_probs.max(dim=-1).values
        accepted_mask = refine_mask.clone()
        accepted_mask &= gpt2_conf.lt(float(config.get("tau_gpt2", 0.5)))
        accepted_mask &= mdlm_conf.gt(float(config.get("tau_mdlm", 0.4)))
        accepted_mask &= (mdlm_conf - gpt2_conf).gt(float(config.get("tau_margin", 0.05)))
        gate_decision = accepted_mask
    elif mode == "candidate_rerank_learned_gate":
        if gate is None:
            raise ValueError("candidate_rerank_learned_gate requires --gate_ckpt")
        accept_prob = torch.sigmoid(gate(feat["features"]))
        accepted_mask = refine_mask & accept_prob.gt(float(config.get("accept_threshold", 0.5)))
        gate_decision = accepted_mask
    else:
        raise ValueError(f"Unknown rerank mode: {mode}")

    rows = accepted_mask.nonzero(as_tuple=False)
    if rows.numel() > 0:
        current = final_logits[rows[:, 0], rows[:, 1]]
        chosen = rerank_token[accepted_mask]
        boost = current.max(dim=-1).values + 1.0
        final_logits[rows[:, 0], rows[:, 1], chosen] = boost
    return final_logits, latency, accepted_mask, feat["candidate_hit_mask"], gate_decision, feat["accept_label"], feat["trainable_mask"]


def finalize(mode: str, ratio: float, top_k: int, acc: dict) -> dict:
    tokens = max(int(acc["tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    correction_rate = acc["corrected"] / max(acc["gpt2_wrong_selected"], 1)
    regression_rate = acc["regressed"] / max(acc["gpt2_correct_selected"], 1)
    return {
        "mode": mode,
        "refine_ratio": ratio,
        "candidate_top_k": top_k,
        "top1": acc["top1_sum"] / tokens,
        "top5": acc["top5_sum"] / tokens,
        "ppl": float(math.exp(min(loss, 50.0))),
        "correction_rate": correction_rate,
        "regression_rate": regression_rate,
        "net_correction": correction_rate - regression_rate,
        "accepted_ratio": acc["accepted"] / max(acc["refined_tokens"], 1),
        "gate_accuracy": acc["gate_correct"] / max(acc["gate_total"], 1) if acc["gate_total"] else "",
        "error_detection_precision": acc["selected_errors"] / max(acc["refined_tokens"], 1),
        "error_detection_recall": acc["selected_errors"] / max(acc["gpt2_errors"], 1),
        "candidate_coverage": acc["candidate_hits"] / max(acc["candidate_total"], 1) if acc["candidate_total"] else "",
        "latency": acc["latency"] / max(acc["batches"], 1),
        "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
    }


@torch.no_grad()
def evaluate(gpt2_model, mdlm_model, gate, val_loader, config, tokenizer_info, device: torch.device) -> list[dict]:
    ratios = [float(value) for value in config.get("refine_ratios", [0.2, 0.3])]
    top_k = int(config.get("candidate_top_k", 20))
    eval_steps = int(config.get("eval_steps", 1000))
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    log_every = int(config.get("log_every", 10))
    modes = [
        "random_refine",
        "candidate_rerank_no_gate",
        "candidate_rerank_rule_gate",
        "candidate_rerank_learned_gate",
    ]
    accs = {("gpt2_only", 0.0): new_acc(), ("mdlm_only", 1.0): new_acc()}
    for ratio in ratios:
        for mode in modes:
            if mode == "candidate_rerank_learned_gate" and gate is None:
                continue
            accs[(mode, ratio)] = new_acc()

    for step, batch in enumerate(val_loader):
        if step >= eval_steps:
            break
        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0):
            print(f"learned_gate_eval_step={step + 1}/{eval_steps}", flush=True)
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
        add_gpt2_only_metrics(accs[("gpt2_only", 0.0)], gpt2_logits, clean, valid_mask, gpt2_latency)

        noised, target_mask = mask_tokens(
            clean,
            int(tokenizer_info.mask_token_id),
            int(tokenizer_info.pad_token_id),
            float(config.get("mask_ratio", 0.15)),
        )
        timesteps = torch.full((clean.size(0),), float(config.get("mask_ratio", 0.15)), device=device)
        sync_if_cuda(device)
        mdlm_start = now()
        mdlm_logits = mdlm_model(noised, timesteps)["logits"].float()
        sync_if_cuda(device)
        add_mdlm_only_metrics(accs[("mdlm_only", 1.0)], mdlm_logits, clean, target_mask, now() - mdlm_start)

        for ratio in ratios:
            uncertainty_mask = select_by_uncertainty(uncertainty, valid_mask, ratio)
            random_mask = select_random(valid_mask, ratio)
            for mode, selected_mask in [
                ("random_refine", random_mask),
                ("candidate_rerank_no_gate", uncertainty_mask),
                ("candidate_rerank_rule_gate", uncertainty_mask),
                ("candidate_rerank_learned_gate", uncertainty_mask),
            ]:
                if mode == "candidate_rerank_learned_gate" and gate is None:
                    continue
                context_mask = expand_refine_window(selected_mask, valid_mask, int(config.get("refine_window", 0)))
                final_logits, latency, accepted_mask, candidate_hit, gate_decision, gate_label, gate_mask = rerank_refine(
                    mdlm_model,
                    gate,
                    draft,
                    gpt2_logits,
                    clean,
                    selected_mask,
                    context_mask,
                    tokenizer_info,
                    device,
                    ratio,
                    mode,
                    config,
                )
                acc = accs[(mode, ratio)]
                add_refined_metrics(
                    acc,
                    final_logits,
                    clean,
                    valid_mask,
                    selected_mask,
                    accepted_mask,
                    gpt2_pred,
                    candidate_hit,
                    gate_decision,
                    gate_label,
                    gate_mask,
                )
                acc["latency"] += gpt2_latency + latency
                acc["batches"] += 1

    rows = [
        finalize("gpt2_only", 0.0, top_k, accs[("gpt2_only", 0.0)]),
        finalize("mdlm_only", 1.0, top_k, accs[("mdlm_only", 1.0)]),
    ]
    for ratio in ratios:
        for mode in modes:
            if (mode, ratio) in accs:
                rows.append(finalize(mode, ratio, top_k, accs[(mode, ratio)]))
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in EVAL_COLUMNS})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict], config: dict, gpt2_source: str) -> None:
    gpt2 = next(row for row in rows if row["mode"] == "gpt2_only")
    learned = [row for row in rows if row["mode"] == "candidate_rerank_learned_gate"]
    no_gate = [row for row in rows if row["mode"] == "candidate_rerank_no_gate"]
    rule = [row for row in rows if row["mode"] == "candidate_rerank_rule_gate"]
    random_rows = [row for row in rows if row["mode"] == "random_refine"]
    best_learned = max(learned, key=lambda row: (row["top5"], row["top1"], row["net_correction"])) if learned else None

    def peer(peer_rows: list[dict], row: dict | None):
        if row is None:
            return None
        return next((item for item in peer_rows if item["refine_ratio"] == row["refine_ratio"]), None)

    no_gate_peer = peer(no_gate, best_learned)
    rule_peer = peer(rule, best_learned)
    random_peer = peer(random_rows, best_learned)
    coverage = best_learned["candidate_coverage"] if best_learned and best_learned["candidate_coverage"] != "" else 0.0
    lines = [
        "# Learned Accept Gate Evaluation",
        "",
        f"- Device GPT-2: `{gpt2_source}`",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- refine_window: `{int(config.get('refine_window', 0))}`",
        f"- refine_ratios: `{config.get('refine_ratios', [0.2, 0.3])}`",
        f"- candidate_top_k: `{int(config.get('candidate_top_k', 20))}`",
        "",
        "## Results",
        "",
        "| Mode | Ratio | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted | Gate Acc | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gate_acc = row["gate_accuracy"] if row["gate_accuracy"] != "" else 0.0
        coverage_row = row["candidate_coverage"] if row["candidate_coverage"] != "" else 0.0
        lines.append(
            f"| {row['mode']} | {row['refine_ratio']:.2f} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['ppl']:.4f} | {row['correction_rate']:.4f} | {row['regression_rate']:.4f} | "
            f"{row['net_correction']:.4f} | {row['accepted_ratio']:.4f} | {gate_acc:.4f} | {coverage_row:.4f} |"
        )
    lines.extend(["", "## Answers", ""])
    if best_learned is None:
        lines.append("1. learned gate 是否降低 regression？未评估，缺少 learned gate checkpoint。")
    else:
        reduces_vs_no_gate = no_gate_peer is not None and best_learned["regression_rate"] < no_gate_peer["regression_rate"]
        keeps_top = best_learned["top1"] >= gpt2["top1"] and best_learned["top5"] >= gpt2["top5"]
        beats_rule = rule_peer is not None and (
            best_learned["net_correction"],
            best_learned["top5"],
        ) > (
            rule_peer["net_correction"],
            rule_peer["top5"],
        )
        beats_no_gate = no_gate_peer is not None and (
            best_learned["net_correction"],
            best_learned["top5"],
        ) > (
            no_gate_peer["net_correction"],
            no_gate_peer["top5"],
        )
        beats_random = random_peer is not None and best_learned["top5"] > random_peer["top5"]
        hypothesis = keeps_top and beats_random and best_learned["net_correction"] >= -0.01
        lines.extend(
            [
                f"1. learned gate 是否降低 regression？{'是' if reduces_vs_no_gate else '否或证据不足'}。",
                f"2. learned gate 是否保持或提升 Top1/Top5？{'是' if keeps_top else '否'}。",
                f"3. learned gate 是否优于 rule-based gate？{'是' if beats_rule else '否或证据不足'}。",
                f"4. learned gate 是否优于 no-gate candidate rerank？{'是' if beats_no_gate else '否或证据不足'}。",
                f"5. candidate coverage 是否足够？{'是' if coverage >= 0.8 else '否或需要尝试 candidate_top_k=50'}；coverage={coverage:.6f}。",
                f"6. 当前结果是否支持“端侧 GPT-2 draft + 边侧 MDLM verifier/refiner”的研究假设？{'支持' if hypothesis else '暂不充分支持'}。",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    display = [
        {
            "Mode": row["mode"],
            "Ratio": row["refine_ratio"],
            "Top1": row["top1"],
            "Top5": row["top5"],
            "PPL": row["ppl"],
            "Correction": row["correction_rate"],
            "Regression": row["regression_rate"],
            "Net": row["net_correction"],
            "Accepted": row["accepted_ratio"],
            "Gate Acc": row["gate_accuracy"],
            "Coverage": row["candidate_coverage"],
        }
        for row in rows
    ]
    print(format_table(display, ["Mode", "Ratio", "Top1", "Top5", "PPL", "Correction", "Regression", "Net", "Accepted", "Gate Acc", "Coverage"]))


def run(config_path: str, gate_ckpt: str | None, eval_steps: int | None = None, save_dir: str | None = None) -> list[dict]:
    config = load_config(config_path)
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)
    if save_dir is not None:
        config["save_dir"] = save_dir
    config["refine_window"] = int(config.get("refine_window", 0))
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    mdlm_model = load_mdlm(config, tokenizer_info, device, resolve_mdlm_checkpoint(config.get("mdlm_ckpt")))
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    gate = load_gate_checkpoint(gate_ckpt, config, device) if gate_ckpt else None
    rows = evaluate(gpt2_model, mdlm_model, gate, val_loader, config, tokenizer_info, device)
    out_dir = Path(config["save_dir"])
    save_csv(out_dir / "learned_gate_eval.csv", rows)
    save_json(out_dir / "learned_gate_eval.json", {"benchmark": rows})
    write_summary(out_dir / "learned_gate_summary.md", rows, config, gpt2_source)
    print_table(rows)
    print(f"saved_learned_gate_eval={out_dir / 'learned_gate_eval.csv'}")
    print(f"saved_learned_gate_summary={out_dir / 'learned_gate_summary.md'}")
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
