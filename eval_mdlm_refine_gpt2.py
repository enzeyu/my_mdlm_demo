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

from data_real import build_dataloaders, mask_tokens
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
    "refined_token_ratio",
    "refined_token_top1",
    "refined_token_top5",
    "correction_rate",
    "regression_rate",
    "error_detection_precision",
    "error_detection_recall",
    "gpt2_error_detection_precision",
    "gpt2_error_detection_recall",
    "net_correction",
    "mdlm_standard_top1",
    "mdlm_standard_top5",
    "mdlm_draft_context_top1",
    "mdlm_draft_context_top5",
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
    config.setdefault("mask_ratio", 0.15)
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
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")
    if ckpt_path is not None and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ("model_state", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        if state_dict is None:
            raise KeyError(f"Could not find a model state dict in checkpoint: {ckpt_path}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"checkpoint_load_warning missing={len(missing)} unexpected={len(unexpected)}")
        print(f"checkpoint_load_status=loaded path={ckpt_path}")
    elif ckpt_path is not None:
        print(f"checkpoint_load_skipped missing_path={ckpt_path}")
    model.eval()
    return model


def validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device: torch.device, max_length: int) -> None:
    """Print and assert the tokenizer/logit surfaces used by the evaluator."""
    sample = torch.full((1, min(8, max_length)), int(tokenizer_info.mask_token_id), device=device, dtype=torch.long)
    timesteps = torch.ones((1,), device=device)
    with torch.no_grad():
        mdlm_vocab = int(mdlm_model(sample, timesteps)["logits"].shape[-1])
    tokenizer_vocab = int(tokenizer_info.vocab_size)
    gpt2_vocab = int(gpt2_model.config.vocab_size)
    print(
        "eval_surface_check "
        f"tokenizer_vocab={tokenizer_vocab} gpt2_vocab={gpt2_vocab} mdlm_logits_vocab={mdlm_vocab} "
        f"pad_token_id={int(tokenizer_info.pad_token_id)} mask_token_id={int(tokenizer_info.mask_token_id)} "
        f"tokenizer_mask_id={tokenizer.mask_token_id}"
    )
    if mdlm_vocab != tokenizer_vocab:
        raise ValueError(f"MDLM logits vocab={mdlm_vocab} does not match tokenizer vocab={tokenizer_vocab}")
    if int(tokenizer_info.mask_token_id) >= tokenizer_vocab:
        raise ValueError(f"mask_token_id={tokenizer_info.mask_token_id} is outside tokenizer vocab={tokenizer_vocab}")


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
    correction_rate = acc["corrected"] / max(acc["gpt2_wrong_selected"], 1)
    regression_rate = acc["regressed"] / max(acc["gpt2_correct_selected"], 1)
    error_precision = acc["selected_errors"] / max(acc["refined_tokens"], 1)
    error_recall = acc["selected_errors"] / max(acc["gpt2_errors"], 1)
    return {
        "mode": mode,
        "refine_ratio": ratio,
        "loss": loss,
        "perplexity": float(math.exp(min(loss, 50.0))),
        "top1_acc": acc["top1_sum"] / tokens,
        "top5_acc": acc["top5_sum"] / tokens,
        "refined_token_ratio": refined_ratio,
        "refined_token_top1": acc["refined_top1_sum"] / refined_tokens,
        "refined_token_top5": acc["refined_top5_sum"] / refined_tokens,
        "correction_rate": correction_rate,
        "regression_rate": regression_rate,
        "error_detection_precision": error_precision,
        "error_detection_recall": error_recall,
        "gpt2_error_detection_precision": error_precision,
        "gpt2_error_detection_recall": error_recall,
        "net_correction": correction_rate - regression_rate,
        "mdlm_standard_top1": "",
        "mdlm_standard_top5": "",
        "mdlm_draft_context_top1": "",
        "mdlm_draft_context_top5": "",
        "estimated_mdlm_compute_saved": 1.0 - refined_ratio,
        "latency": acc["latency"] / max(acc["batches"], 1),
        "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
    }


def attach_diagnostic_metrics(rows: list[dict]) -> None:
    standard = next(row for row in rows if row["mode"] == "mdlm_only")
    draft_rows = [row for row in rows if row["mode"] == "gpt2_mdlm_refine"]
    best_draft = max(draft_rows, key=lambda row: row["top5_acc"]) if draft_rows else None
    for row in rows:
        row["mdlm_standard_top1"] = standard["top1_acc"]
        row["mdlm_standard_top5"] = standard["top5_acc"]
        if best_draft is not None:
            row["mdlm_draft_context_top1"] = best_draft["refined_token_top1"]
            row["mdlm_draft_context_top5"] = best_draft["refined_token_top5"]


@torch.no_grad()
def evaluate(gpt2_model, mdlm_model, val_loader, config, tokenizer_info, device: torch.device):
    ratios = [float(value) for value in config.get("refine_ratios", [0.05, 0.1, 0.2, 0.3])]
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    eval_steps = int(config["eval_steps"])
    log_every = int(config.get("log_every", 10))
    accumulators: dict[tuple[str, float], dict] = {
        ("gpt2_only", 0.0): new_acc(),
        ("mdlm_only", 1.0): new_acc(),
    }
    for ratio in ratios:
        accumulators[("random_refine", ratio)] = new_acc()
        accumulators[("gpt2_mdlm_refine", ratio)] = new_acc()

    for step, batch in enumerate(val_loader):
        if step >= eval_steps:
            break
        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0):
            print(f"eval_step={step + 1}/{eval_steps}", flush=True)
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
        mdlm_latency = now() - mdlm_start
        add_mdlm_only_metrics(accumulators[("mdlm_only", 1.0)], mdlm_logits, clean, target_mask, mdlm_latency)

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
    attach_diagnostic_metrics(rows)
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
    correction_ok = best_refine["correction_rate"] > best_refine["regression_rate"]
    standard_top5 = float(mdlm["mdlm_standard_top5"])
    draft_top5 = float(best_refine["refined_token_top5"])
    draft_context_hurts = standard_top5 > draft_top5 + 0.05
    standard_is_weak = standard_top5 < 0.2

    lines = [
        "# Diffusion-Assisted Autoregressive Refinement",
        "",
        f"- Device GPT-2: `{gpt2_source}`",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- Eval steps: `{int(config.get('eval_steps', 0))}`",
        f"- Mask ratio: `{float(config.get('mask_ratio', 0.15))}`",
        f"- Uncertainty score: `{config.get('uncertainty_score', 'inverse_confidence')}`",
        f"- Refine ratios: `{config.get('refine_ratios', [0.05, 0.1, 0.2, 0.3])}`",
        "",
        "| Mode | Refine Ratio | Top1 | Top5 | PPL | Correction | Regression | Net Correction | Error Detect Precision | Error Detect Recall | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['refine_ratio']:.2f} | {row['top1_acc']:.4f} | "
            f"{row['top5_acc']:.4f} | {row['perplexity']:.4f} | {row['correction_rate']:.4f} | "
            f"{row['regression_rate']:.4f} | {row['net_correction']:.4f} | {row['error_detection_precision']:.4f} | "
            f"{row['error_detection_recall']:.4f} | {row['latency']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- mdlm_standard_top1={mdlm['mdlm_standard_top1']:.6f}",
            f"- mdlm_standard_top5={mdlm['mdlm_standard_top5']:.6f}",
            f"- mdlm_draft_context_top1={best_refine['mdlm_draft_context_top1']:.6f}",
            f"- mdlm_draft_context_top5={best_refine['mdlm_draft_context_top5']:.6f}",
            f"- best_gpt2_mdlm_refine_ratio={best_refine['refine_ratio']:.2f}",
            f"- best_gpt2_error_detection_precision={best_refine['gpt2_error_detection_precision']:.6f}",
            f"- best_gpt2_error_detection_recall={best_refine['gpt2_error_detection_recall']:.6f}",
            "",
            "## Questions",
            "",
            f"1. MDLM-only baseline 为什么异常低？旧逻辑把整段序列全部 mask，并在 full valid sequence 上评估，和 edge_only 的随机 masked recovery 不一致；当前已改为按照 mask_ratio 构造 target_mask，并只在 masked positions 上统计。",
            f"2. 是 evaluation bug，还是 GPT-2 draft context 导致 MDLM 失效？{'standard masked recovery 仍偏弱，更像 loading/mask/eval 仍需继续查' if standard_is_weak else ('standard recovery 明显强于 draft-context recovery，主要问题是 GPT-2 draft context 误导 MDLM' if draft_context_hurts else '当前诊断未显示 standard 与 draft context 有巨大断层')}。",
            f"3. eval_steps 从 200 增加到 1000 后，结果是否稳定？本次配置 eval_steps={int(config.get('eval_steps', 0))}；稳定性需和 200-step 旧结果对比，重点看 Top1/Top5 排序是否保持。",
            f"4. gpt2_mdlm_refine 是否稳定优于 gpt2_only？{'是' if beats_gpt2 else '否，或证据不足'}。最佳 refine Top5 gain={best_refine['top5_acc'] - gpt2['top5_acc']:.6f}，Top1 gain={best_refine['top1_acc'] - gpt2['top1_acc']:.6f}。",
            f"5. uncertainty selection 是否稳定优于 random refine？{'是' if beats_random else '否'}。同 ratio 下 Top5 gain={best_refine['top5_acc'] - same_ratio_random['top5_acc']:.6f}。",
            f"6. 下一步是否应该加入 accept gate，而不是直接替换 GPT-2 token？{'是' if not correction_ok else '仍建议加入 gate 做风险控制'}。best correction_rate={best_refine['correction_rate']:.6f}，regression_rate={best_refine['regression_rate']:.6f}，net_correction={best_refine['net_correction']:.6f}。",
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
            "Net Correction": row["net_correction"],
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
                "Net Correction",
                "Error Detect Precision",
                "Error Detect Recall",
                "Latency",
            ],
        )
    )


def run(config_path: str, mdlm_ckpt: str | None, eval_steps: int | None = None, save_dir: str | None = None) -> list[dict]:
    config = load_config(config_path)
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)
    if save_dir is not None:
        config["save_dir"] = save_dir
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    mdlm_model = load_mdlm(config, tokenizer_info, device, resolve_mdlm_checkpoint(mdlm_ckpt))
    validate_model_surfaces(mdlm_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    rows = evaluate(gpt2_model, mdlm_model, val_loader, config, tokenizer_info, device)
    save_dir = Path(config["save_dir"])
    eval_stem = str(config.get("eval_output_stem", "gpt2_mdlm_refine_eval"))
    summary_name = str(config.get("summary_output_name", "gpt2_mdlm_refine_summary.md"))
    save_csv(save_dir / f"{eval_stem}.csv", rows)
    save_json(save_dir / f"{eval_stem}.json", {"benchmark": rows})
    write_summary(save_dir / summary_name, rows, config, gpt2_source)
    print_table(rows)
    print(f"saved_refine={save_dir / f'{eval_stem}.csv'}")
    print(f"saved_refine_summary={save_dir / summary_name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mdlm_ckpt", default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    run(args.config, args.mdlm_ckpt, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
