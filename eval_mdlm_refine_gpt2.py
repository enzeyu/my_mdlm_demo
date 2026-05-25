"""Evaluate Diffusion-Assisted Autoregressive Refinement.

GPT-2 small produces left-to-right draft predictions.  MDLM then refines only
low-confidence draft positions in parallel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from data_real import build_dataloaders
from eval_gpt2_assist import check_tokenizer_compatibility, choose_device
from metrics import format_table, now, sync_if_cuda
from models_mdlm_wrapper import build_edge_mdlm_model


EVAL_COLUMNS = [
    "mode",
    "refine_ratio",
    "loss",
    "perplexity",
    "top1_acc",
    "top5_acc",
    "refined_token_top1",
    "refined_token_top5",
    "correction_rate",
    "regression_rate",
    "error_detection_precision",
    "error_detection_recall",
    "estimated_mdlm_compute_saved",
    "latency",
    "tokens_per_sec",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "edge_model_name_or_path" in config and "pretrained_edge_path" not in config:
        config["pretrained_edge_path"] = config["edge_model_name_or_path"]
    config.setdefault("dataset_cache_dir", "/mnt/data/enzeyu/hf_downloads/datasets")
    config.setdefault("hf_local_files_only", True)
    config.setdefault("uncertainty_score", "inverse_confidence")
    config.setdefault("refine_ratios", [0.05, 0.1, 0.2, 0.3])
    config.setdefault("gpt2_query_batch_size", 0)
    return config


def resolve_mdlm_checkpoint(path: str | None) -> Path | None:
    return Path(path) if path else None


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
    return model.to(device), model_path


def load_mdlm(config: dict, tokenizer_info, device: torch.device, ckpt_path: Path | None):
    model = build_edge_mdlm_model(config, tokenizer_info.vocab_size, tokenizer_info.pad_token_id, tokenizer_info.mask_token_id).to(device)
    if ckpt_path is not None and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing or unexpected:
            print(f"checkpoint_load_warning missing={len(missing)} unexpected={len(unexpected)}")
    elif ckpt_path is not None:
        print(f"checkpoint_load_skipped missing_path={ckpt_path}")
    model.eval()
    return model


@torch.no_grad()
def gpt2_teacher_forced_logits(gpt2_model, clean: torch.Tensor, eos_token_id: int):
    """Predict x_i from [eos, x_0, ..., x_{i-1}] for every position."""
    shifted = clean.new_empty(clean.shape)
    shifted[:, 0] = eos_token_id
    shifted[:, 1:] = clean[:, :-1]
    attention_mask = torch.ones_like(shifted)
    outputs = gpt2_model(input_ids=shifted, attention_mask=attention_mask)
    return outputs.logits.float()


def uncertainty_from_logits(logits: torch.Tensor, score_name: str) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    if score_name == "entropy":
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return -(probs * log_probs).sum(dim=-1)
    if score_name == "margin":
        top2 = probs.topk(2, dim=-1).values
        return 1.0 - (top2[..., 0] - top2[..., 1])
    return 1.0 - probs.max(dim=-1).values


def select_by_uncertainty(uncertainty: torch.Tensor, valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    coords = valid_mask.nonzero(as_tuple=False)
    count = int(coords.size(0))
    if count == 0 or ratio <= 0.0:
        return selected
    k = min(count, max(1, int(math.ceil(count * ratio))))
    scores = uncertainty[valid_mask]
    chosen = scores.topk(k, largest=True).indices
    chosen_coords = coords[chosen]
    selected[chosen_coords[:, 0], chosen_coords[:, 1]] = True
    return selected


def select_random(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    coords = valid_mask.nonzero(as_tuple=False)
    count = int(coords.size(0))
    if count == 0 or ratio <= 0.0:
        return selected
    k = min(count, max(1, int(math.ceil(count * ratio))))
    chosen = torch.randperm(count, device=valid_mask.device)[:k]
    chosen_coords = coords[chosen]
    selected[chosen_coords[:, 0], chosen_coords[:, 1]] = True
    return selected


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
    gpt2_pred: torch.Tensor,
) -> None:
    add_logits_metrics(acc, final_logits, labels, valid_mask)
    gpt2_correct_all = gpt2_pred.eq(labels) & valid_mask
    gpt2_wrong_all = ~gpt2_pred.eq(labels) & valid_mask
    acc["gpt2_errors"] += int(gpt2_wrong_all.sum().item())

    if not refine_mask.any():
        return
    refined_logits = final_logits[refine_mask]
    refined_labels = labels[refine_mask]
    refined_top1 = refined_logits.argmax(dim=-1)
    refined_top5 = refined_logits.topk(min(5, refined_logits.size(-1)), dim=-1).indices.eq(refined_labels.unsqueeze(-1)).any(dim=-1)
    acc["refined_top1_sum"] += int(refined_top1.eq(refined_labels).sum().item())
    acc["refined_top5_sum"] += int(refined_top5.sum().item())
    acc["refined_tokens"] += int(refined_labels.numel())

    gpt2_correct = gpt2_correct_all[refine_mask]
    gpt2_wrong = gpt2_wrong_all[refine_mask]
    refined_correct = refined_top1.eq(refined_labels)
    acc["corrected"] += int((gpt2_wrong & refined_correct).sum().item())
    acc["regressed"] += int((gpt2_correct & ~refined_correct).sum().item())
    acc["gpt2_wrong_selected"] += int(gpt2_wrong.sum().item())
    acc["gpt2_correct_selected"] += int(gpt2_correct.sum().item())
    acc["selected_errors"] += int(gpt2_wrong.sum().item())


def add_gpt2_only_metrics(acc: dict, gpt2_logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, latency: float) -> None:
    add_logits_metrics(acc, gpt2_logits, labels, valid_mask)
    gpt2_pred = gpt2_logits.argmax(dim=-1)
    acc["gpt2_errors"] += int((~gpt2_pred.eq(labels) & valid_mask).sum().item())
    acc["latency"] += latency
    acc["batches"] += 1


def add_mdlm_only_metrics(acc: dict, mdlm_logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, latency: float) -> None:
    top1_before = acc["top1_sum"]
    top5_before = acc["top5_sum"]
    tokens_before = acc["tokens"]
    add_logits_metrics(acc, mdlm_logits, labels, valid_mask)
    acc["refined_top1_sum"] += acc["top1_sum"] - top1_before
    acc["refined_top5_sum"] += acc["top5_sum"] - top5_before
    acc["refined_tokens"] += acc["tokens"] - tokens_before
    acc["latency"] += latency
    acc["batches"] += 1


def pad_gpt2_logits(gpt2_logits: torch.Tensor, target_vocab: int) -> torch.Tensor:
    if gpt2_logits.size(-1) == target_vocab:
        return gpt2_logits
    padded = gpt2_logits.new_full((*gpt2_logits.shape[:-1], target_vocab), -1e4)
    padded[..., : gpt2_logits.size(-1)] = gpt2_logits
    return padded


@torch.no_grad()
def refine_once(
    mdlm_model,
    draft: torch.Tensor,
    gpt2_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    refine_mask: torch.Tensor,
    tokenizer_info,
    device: torch.device,
    ratio: float,
):
    masked_draft = draft.clone()
    masked_draft[refine_mask] = tokenizer_info.mask_token_id
    timesteps = torch.full((draft.size(0),), max(float(ratio), 1e-4), device=device)
    sync_if_cuda(device)
    start = now()
    mdlm_logits = mdlm_model(masked_draft, timesteps)["logits"].float()
    sync_if_cuda(device)
    latency = now() - start
    final_logits = pad_gpt2_logits(gpt2_logits, mdlm_logits.size(-1)).clone()
    final_logits[refine_mask] = mdlm_logits[refine_mask]
    return final_logits, latency


def finalize(mode: str, ratio: float, acc: dict) -> dict:
    tokens = max(int(acc["tokens"]), 1)
    refined_tokens = max(int(acc["refined_tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    refined_ratio = acc["refined_tokens"] / tokens
    return {
        "mode": mode,
        "refine_ratio": ratio,
        "loss": loss,
        "perplexity": float(math.exp(min(loss, 50.0))),
        "top1_acc": acc["top1_sum"] / tokens,
        "top5_acc": acc["top5_sum"] / tokens,
        "refined_token_top1": acc["refined_top1_sum"] / refined_tokens,
        "refined_token_top5": acc["refined_top5_sum"] / refined_tokens,
        "correction_rate": acc["corrected"] / max(acc["gpt2_wrong_selected"], 1),
        "regression_rate": acc["regressed"] / max(acc["gpt2_correct_selected"], 1),
        "error_detection_precision": acc["selected_errors"] / max(acc["refined_tokens"], 1),
        "error_detection_recall": acc["selected_errors"] / max(acc["gpt2_errors"], 1),
        "estimated_mdlm_compute_saved": 1.0 - refined_ratio,
        "latency": acc["latency"] / max(acc["batches"], 1),
        "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
    }


@torch.no_grad()
def evaluate(gpt2_model, mdlm_model, val_loader, config, tokenizer_info, device: torch.device):
    ratios = [float(value) for value in config.get("refine_ratios", [0.05, 0.1, 0.2, 0.3])]
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    accumulators: dict[tuple[str, float], dict] = {
        ("gpt2_only", 0.0): new_acc(),
        ("mdlm_only", 1.0): new_acc(),
    }
    for ratio in ratios:
        accumulators[("random_refine", ratio)] = new_acc()
        accumulators[("gpt2_mdlm_refine", ratio)] = new_acc()

    for step, batch in enumerate(val_loader):
        if step >= int(config["eval_steps"]):
            break
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
        add_gpt2_only_metrics(accumulators[("gpt2_only", 0.0)], gpt2_logits, clean, valid_mask, gpt2_latency)

        full_masked = clean.new_full(clean.shape, int(tokenizer_info.mask_token_id))
        timesteps = torch.ones((clean.size(0),), device=device)
        sync_if_cuda(device)
        mdlm_start = now()
        mdlm_logits = mdlm_model(full_masked, timesteps)["logits"].float()
        sync_if_cuda(device)
        mdlm_latency = now() - mdlm_start
        add_mdlm_only_metrics(accumulators[("mdlm_only", 1.0)], mdlm_logits, clean, valid_mask, mdlm_latency)

        for ratio in ratios:
            uncertainty_mask = select_by_uncertainty(uncertainty, valid_mask, ratio)
            random_mask = select_random(valid_mask, ratio)
            for mode, refine_mask in [("random_refine", random_mask), ("gpt2_mdlm_refine", uncertainty_mask)]:
                final_logits, refine_latency = refine_once(
                    mdlm_model,
                    draft,
                    gpt2_logits,
                    clean,
                    valid_mask,
                    refine_mask,
                    tokenizer_info,
                    device,
                    ratio,
                )
                acc = accumulators[(mode, ratio)]
                add_refined_metrics(acc, final_logits, clean, valid_mask, refine_mask, gpt2_pred)
                acc["latency"] += gpt2_latency + refine_latency
                acc["batches"] += 1

    rows = [finalize("gpt2_only", 0.0, accumulators[("gpt2_only", 0.0)]), finalize("mdlm_only", 1.0, accumulators[("mdlm_only", 1.0)])]
    for ratio in ratios:
        rows.append(finalize("random_refine", ratio, accumulators[("random_refine", ratio)]))
        rows.append(finalize("gpt2_mdlm_refine", ratio, accumulators[("gpt2_mdlm_refine", ratio)]))
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
    mdlm = next(row for row in rows if row["mode"] == "mdlm_only")
    refine_rows = [row for row in rows if row["mode"] == "gpt2_mdlm_refine"]
    random_rows = [row for row in rows if row["mode"] == "random_refine"]
    best_refine = max(refine_rows, key=lambda row: row["top5_acc"])
    same_ratio_random = next(row for row in random_rows if row["refine_ratio"] == best_refine["refine_ratio"])
    beats_gpt2 = best_refine["top5_acc"] > gpt2["top5_acc"] and best_refine["top1_acc"] > gpt2["top1_acc"]
    beats_random = best_refine["top5_acc"] > same_ratio_random["top5_acc"]
    close_to_mdlm = (mdlm["top5_acc"] - best_refine["top5_acc"]) <= 0.02
    correction_ok = best_refine["correction_rate"] > best_refine["regression_rate"]

    lines = [
        "# Diffusion-Assisted Autoregressive Refinement",
        "",
        f"- Device GPT-2: `{gpt2_source}`",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- Uncertainty score: `{config.get('uncertainty_score', 'inverse_confidence')}`",
        f"- Refine ratios: `{config.get('refine_ratios', [0.05, 0.1, 0.2, 0.3])}`",
        "",
        "| Mode | Refine Ratio | Top1 | Top5 | PPL | Correction | Regression | Error Detect Precision | Error Detect Recall | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['refine_ratio']:.2f} | {row['top1_acc']:.4f} | "
            f"{row['top5_acc']:.4f} | {row['perplexity']:.4f} | {row['correction_rate']:.4f} | "
            f"{row['regression_rate']:.4f} | {row['error_detection_precision']:.4f} | "
            f"{row['error_detection_recall']:.4f} | {row['latency']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"1. MDLM 是否能有效修正 GPT-2 的低置信 token？{'是' if beats_gpt2 and correction_ok else '否，或证据不足'}。最佳 refine ratio={best_refine['refine_ratio']:.2f}，Top5 gain={best_refine['top5_acc'] - gpt2['top5_acc']:.6f}。",
            f"2. uncertainty selection 是否优于 random selection？{'是' if beats_random else '否'}。同 ratio 下 Top5 gain={best_refine['top5_acc'] - same_ratio_random['top5_acc']:.6f}。",
            f"3. 只 refine 少量 token 是否能获得明显质量提升？{'是' if beats_gpt2 else '否'}。最佳 refine 的 estimated compute saved={best_refine['estimated_mdlm_compute_saved']:.4f}。",
            f"4. 当前方向是否比 GPT-2 assist MDLM 更有希望？{'是' if beats_gpt2 else '暂时还不能证明'}。这个方向至少把强 MDLM 用在弱 GPT-2 的错误位置，建模方向更合理。",
            f"5. 下一步建议：先优化 error detection，尝试 threshold/gating、校准 GPT-2 confidence、加入 right-context verifier 特征；若离 full MDLM 仍有差距，再训练一个轻量 selector 或 distill MDLM refinement policy。",
            "",
            f"Full MDLM Top5={mdlm['top5_acc']:.6f}；最佳 refine Top5={best_refine['top5_acc']:.6f}；是否接近 full MDLM：{'是' if close_to_mdlm else '否'}。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    display = [
        {
            "Mode": row["mode"],
            "Refine Ratio": row["refine_ratio"],
            "Top1": row["top1_acc"],
            "Top5": row["top5_acc"],
            "PPL": row["perplexity"],
            "Correction": row["correction_rate"],
            "Regression": row["regression_rate"],
            "Error Detect Precision": row["error_detection_precision"],
            "Error Detect Recall": row["error_detection_recall"],
            "Latency": row["latency"],
        }
        for row in rows
    ]
    print(
        format_table(
            display,
            [
                "Mode",
                "Refine Ratio",
                "Top1",
                "Top5",
                "PPL",
                "Correction",
                "Regression",
                "Error Detect Precision",
                "Error Detect Recall",
                "Latency",
            ],
        )
    )


def run(config_path: str, mdlm_ckpt: str | None) -> list[dict]:
    config = load_config(config_path)
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    mdlm_model = load_mdlm(config, tokenizer_info, device, resolve_mdlm_checkpoint(mdlm_ckpt))
    rows = evaluate(gpt2_model, mdlm_model, val_loader, config, tokenizer_info, device)
    save_dir = Path(config["save_dir"])
    save_csv(save_dir / "gpt2_mdlm_refine_eval.csv", rows)
    save_json(save_dir / "gpt2_mdlm_refine_eval.json", {"benchmark": rows})
    write_summary(save_dir / "gpt2_mdlm_refine_summary.md", rows, config, gpt2_source)
    print_table(rows)
    print(f"saved_refine={save_dir / 'gpt2_mdlm_refine_eval.csv'}")
    print(f"saved_refine_summary={save_dir / 'gpt2_mdlm_refine_summary.md'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mdlm_ckpt", default=None)
    args = parser.parse_args()
    run(args.config, args.mdlm_ckpt)


if __name__ == "__main__":
    main()
