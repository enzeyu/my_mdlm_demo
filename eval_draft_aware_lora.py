"""Evaluate pretrained MDLM and draft-aware LoRA MDLM on GPT-2 draft contexts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data_real import build_dataloaders, mask_tokens
from draft_aware_lora_utils import (
    DEFAULT_LORA_TARGETS,
    block_exact_match,
    build_draft_aware_inputs,
    freeze_module,
    inject_lora,
    load_lora_adapter,
)
from metrics import format_table, gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda
from refine_gate import pad_gpt2_logits
from refine_utils import (
    choose_device,
    gpt2_teacher_forced_logits,
    load_config,
    load_gpt2,
    load_mdlm,
    uncertainty_from_logits,
    validate_model_surfaces,
)


EVAL_COLUMNS = [
    "mode",
    "top1",
    "top5",
    "ppl",
    "draft_context_top1",
    "draft_context_top5",
    "correction_rate",
    "regression_rate",
    "net_correction",
    "selected_token_ratio",
    "block_exact_match",
    "latency",
    "tokens_per_sec",
    "gpu_memory_MB",
]


def new_acc() -> dict:
    return {
        "loss_sum": 0.0,
        "top1_sum": 0,
        "top5_sum": 0,
        "tokens": 0,
        "draft_top1_sum": 0,
        "draft_top5_sum": 0,
        "draft_tokens": 0,
        "selected_tokens": 0,
        "valid_tokens": 0,
        "corrected": 0,
        "regressed": 0,
        "gpt2_wrong_selected": 0,
        "gpt2_correct_selected": 0,
        "block_exact_good": 0,
        "block_exact_total": 0,
        "latency": 0.0,
        "batches": 0,
        "gpu_memory_MB": 0.0,
    }


def add_logits(acc: dict, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, draft_context: bool = False) -> None:
    if not bool(mask.any()):
        return
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    vocab = selected_logits.size(-1)
    loss = F.cross_entropy(selected_logits.view(-1, vocab), selected_labels.view(-1), reduction="sum")
    pred = selected_logits.argmax(dim=-1)
    top5 = selected_logits.topk(min(5, vocab), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
    count = int(selected_labels.numel())
    acc["loss_sum"] += float(loss.item())
    acc["top1_sum"] += int(pred.eq(selected_labels).sum().item())
    acc["top5_sum"] += int(top5.sum().item())
    acc["tokens"] += count
    if draft_context:
        acc["draft_top1_sum"] += int(pred.eq(selected_labels).sum().item())
        acc["draft_top5_sum"] += int(top5.sum().item())
        acc["draft_tokens"] += count


def add_refinement(acc: dict, final_logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, selected: torch.Tensor, gpt2_pred: torch.Tensor, block_size: int) -> None:
    add_logits(acc, final_logits, labels, valid_mask)
    add_logits(acc, final_logits, labels, selected, draft_context=True)
    final_pred = final_logits.argmax(dim=-1)
    gpt2_correct = gpt2_pred.eq(labels) & selected
    gpt2_wrong = ~gpt2_pred.eq(labels) & selected
    final_correct = final_pred.eq(labels)
    acc["corrected"] += int((gpt2_wrong & final_correct).sum().item())
    acc["regressed"] += int((gpt2_correct & ~final_correct).sum().item())
    acc["gpt2_wrong_selected"] += int(gpt2_wrong.sum().item())
    acc["gpt2_correct_selected"] += int(gpt2_correct.sum().item())
    good, total = block_exact_match(final_pred, labels, selected, block_size)
    acc["block_exact_good"] += good
    acc["block_exact_total"] += total


def finalize(mode: str, acc: dict) -> dict:
    tokens = max(int(acc["tokens"]), 1)
    draft_tokens = max(int(acc["draft_tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    correction_rate = acc["corrected"] / max(acc["gpt2_wrong_selected"], 1)
    regression_rate = acc["regressed"] / max(acc["gpt2_correct_selected"], 1)
    return {
        "mode": mode,
        "top1": acc["top1_sum"] / tokens,
        "top5": acc["top5_sum"] / tokens,
        "ppl": float(math.exp(min(loss, 50.0))),
        "draft_context_top1": acc["draft_top1_sum"] / draft_tokens if acc["draft_tokens"] else "",
        "draft_context_top5": acc["draft_top5_sum"] / draft_tokens if acc["draft_tokens"] else "",
        "correction_rate": correction_rate,
        "regression_rate": regression_rate,
        "net_correction": correction_rate - regression_rate,
        "selected_token_ratio": acc["selected_tokens"] / max(acc["valid_tokens"], 1),
        "block_exact_match": acc["block_exact_good"] / max(acc["block_exact_total"], 1) if acc["block_exact_total"] else "",
        "latency": acc["latency"] / max(acc["batches"], 1),
        "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
        "gpu_memory_MB": acc["gpu_memory_MB"],
    }


@torch.no_grad()
def run_model(model, input_ids: torch.Tensor, ratio: float, device: torch.device) -> tuple[torch.Tensor, float, float]:
    timesteps = torch.full((input_ids.size(0),), max(float(ratio), 1e-4), device=device)
    reset_gpu_memory(device)
    sync_if_cuda(device)
    start = now()
    logits = model(input_ids, timesteps)["logits"].float()
    sync_if_cuda(device)
    return logits, now() - start, gpu_memory_mb(device)


@torch.no_grad()
def evaluate(config: dict, lora_path: str | None) -> list[dict]:
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, gpt2_source = load_gpt2(config, tokenizer, device)
    pretrained = load_mdlm(config, tokenizer_info, device, None).to(device)
    validate_model_surfaces(pretrained, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    freeze_module(gpt2_model)
    freeze_module(pretrained)
    pretrained.eval()

    lora_model = None
    if lora_path:
        lora_model = load_mdlm(config, tokenizer_info, device, None).to(device)
        freeze_module(lora_model)
        meta_path = Path(lora_path) / "adapter_config.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        inject_lora(
            lora_model,
            target_names=metadata.get("lora_target_modules", config.get("lora_target_modules", DEFAULT_LORA_TARGETS)),
            r=int(metadata.get("lora_r", config.get("lora_r", 8))),
            alpha=float(metadata.get("lora_alpha", config.get("lora_alpha", 16))),
            dropout=float(metadata.get("lora_dropout", config.get("lora_dropout", 0.05))),
        )
        load_lora_adapter(lora_model, lora_path, device)
        lora_model.eval()

    modes = [
        "gpt2_only",
        "pretrained_mdlm_standard",
        "pretrained_mdlm_draft_context",
        "gpt2_plus_pretrained_mdlm_refine",
    ]
    if lora_model is not None:
        modes.extend(["lora_mdlm_standard", "lora_mdlm_draft_context", "gpt2_plus_lora_mdlm_refine"])
    accs = {mode: new_acc() for mode in modes}
    block_size = int(config.get("block_size", 4))
    refine_ratio = float(config.get("refine_ratio", 0.2))
    score_name = str(config.get("uncertainty_score", "inverse_confidence"))
    eval_steps = int(config.get("eval_steps", 1000))
    log_interval = int(config.get("log_interval", config.get("log_every", 100)))

    for step, batch in enumerate(val_loader):
        if step >= eval_steps:
            break
        if log_interval > 0 and (step == 0 or (step + 1) % log_interval == 0):
            print(f"draft_aware_eval_step={step + 1}/{eval_steps}", flush=True)
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        reset_gpu_memory(device)
        sync_if_cuda(device)
        gpt2_start = now()
        gpt2_logits = gpt2_teacher_forced_logits(gpt2_model, clean, int(tokenizer_info.pad_token_id))
        sync_if_cuda(device)
        gpt2_latency = now() - gpt2_start
        gpt2_pred = gpt2_logits.argmax(dim=-1)
        draft = gpt2_pred.clamp_max(int(tokenizer_info.mask_token_id) - 1)
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
        selected_count = int(selected.sum().item())
        valid_count = int(valid_mask.sum().item())

        acc = accs["gpt2_only"]
        add_logits(acc, gpt2_logits, clean, valid_mask)
        acc["latency"] += gpt2_latency
        acc["batches"] += 1
        acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], gpu_memory_mb(device))
        acc["valid_tokens"] += valid_count

        noised, target_mask = mask_tokens(clean, int(tokenizer_info.mask_token_id), int(tokenizer_info.pad_token_id), float(config.get("mask_ratio", 0.15)))
        std_logits, std_latency, std_mem = run_model(pretrained, noised, float(config.get("mask_ratio", 0.15)), device)
        acc = accs["pretrained_mdlm_standard"]
        add_logits(acc, std_logits, clean, target_mask)
        acc["selected_tokens"] += int(target_mask.sum().item())
        acc["valid_tokens"] += valid_count
        acc["latency"] += std_latency
        acc["batches"] += 1
        acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], std_mem)

        if lora_model is not None:
            lora_std_logits, lora_std_latency, lora_std_mem = run_model(
                lora_model,
                noised,
                float(config.get("mask_ratio", 0.15)),
                device,
            )
            acc = accs["lora_mdlm_standard"]
            add_logits(acc, lora_std_logits, clean, target_mask)
            acc["selected_tokens"] += int(target_mask.sum().item())
            acc["valid_tokens"] += valid_count
            acc["latency"] += lora_std_latency
            acc["batches"] += 1
            acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], lora_std_mem)

        pre_logits, pre_latency, pre_mem = run_model(pretrained, mdlm_input, refine_ratio, device)
        acc = accs["pretrained_mdlm_draft_context"]
        add_logits(acc, pre_logits, clean, selected, draft_context=True)
        good, total = block_exact_match(pre_logits.argmax(dim=-1), clean, selected, block_size)
        acc["block_exact_good"] += good
        acc["block_exact_total"] += total
        acc["selected_tokens"] += selected_count
        acc["valid_tokens"] += valid_count
        acc["latency"] += pre_latency
        acc["batches"] += 1
        acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], pre_mem)

        gpt2_padded = pad_gpt2_logits(gpt2_logits, pre_logits.size(-1))
        final_pre = gpt2_padded.clone()
        final_pre[selected] = pre_logits[selected]
        acc = accs["gpt2_plus_pretrained_mdlm_refine"]
        add_refinement(acc, final_pre, clean, valid_mask, selected, gpt2_pred, block_size)
        acc["selected_tokens"] += selected_count
        acc["valid_tokens"] += valid_count
        acc["latency"] += gpt2_latency + pre_latency
        acc["batches"] += 1
        acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], pre_mem)

        if lora_model is not None:
            lora_logits, lora_latency, lora_mem = run_model(lora_model, mdlm_input, refine_ratio, device)
            acc = accs["lora_mdlm_draft_context"]
            add_logits(acc, lora_logits, clean, selected, draft_context=True)
            good, total = block_exact_match(lora_logits.argmax(dim=-1), clean, selected, block_size)
            acc["block_exact_good"] += good
            acc["block_exact_total"] += total
            acc["selected_tokens"] += selected_count
            acc["valid_tokens"] += valid_count
            acc["latency"] += lora_latency
            acc["batches"] += 1
            acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], lora_mem)

            gpt2_padded = pad_gpt2_logits(gpt2_logits, lora_logits.size(-1))
            final_lora = gpt2_padded.clone()
            final_lora[selected] = lora_logits[selected]
            acc = accs["gpt2_plus_lora_mdlm_refine"]
            add_refinement(acc, final_lora, clean, valid_mask, selected, gpt2_pred, block_size)
            acc["selected_tokens"] += selected_count
            acc["valid_tokens"] += valid_count
            acc["latency"] += gpt2_latency + lora_latency
            acc["batches"] += 1
            acc["gpu_memory_MB"] = max(acc["gpu_memory_MB"], lora_mem)

    print(f"gpt2_source={gpt2_source}", flush=True)
    return [finalize(mode, accs[mode]) for mode in modes]


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EVAL_COLUMNS})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict], config: dict) -> None:
    by_mode = {row["mode"]: row for row in rows}
    pre_draft = by_mode.get("pretrained_mdlm_draft_context")
    lora_draft = by_mode.get("lora_mdlm_draft_context")
    pre_refine = by_mode.get("gpt2_plus_pretrained_mdlm_refine")
    lora_refine = by_mode.get("gpt2_plus_lora_mdlm_refine")
    std = by_mode.get("pretrained_mdlm_standard")

    draft_drop = bool(pre_draft and std and pre_draft["draft_context_top1"] != "" and pre_draft["draft_context_top1"] < std["top1"])
    lora_beats_draft = bool(lora_draft and pre_draft and lora_draft["draft_context_top1"] > pre_draft["draft_context_top1"])
    lora_beats_refine = bool(lora_refine and pre_refine and lora_refine["net_correction"] > pre_refine["net_correction"])
    corr_up = bool(lora_refine and pre_refine and lora_refine["correction_rate"] > pre_refine["correction_rate"])
    reg_down = bool(lora_refine and pre_refine and lora_refine["regression_rate"] < pre_refine["regression_rate"])
    net_good = bool(lora_refine and lora_refine["net_correction"] >= 0)
    hypothesis = lora_beats_draft and lora_beats_refine and (corr_up or reg_down)

    lines = [
        "# Draft-aware LoRA Evaluation",
        "",
        f"- Device GPT-2: `{config.get('device_model_name_or_path')}`",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- block_size: `{int(config.get('block_size', 4))}`",
        f"- refine_ratio: `{float(config.get('refine_ratio', 0.2))}`",
        "",
        "## Results",
        "",
        "| Mode | Top1 | Top5 | PPL | Draft Top1 | Draft Top5 | Correction | Regression | Net | Selected | Block EM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        draft_top1 = row["draft_context_top1"] if row["draft_context_top1"] != "" else 0.0
        draft_top5 = row["draft_context_top5"] if row["draft_context_top5"] != "" else 0.0
        block_em = row["block_exact_match"] if row["block_exact_match"] != "" else 0.0
        lines.append(
            f"| {row['mode']} | {row['top1']:.4f} | {row['top5']:.4f} | {row['ppl']:.4f} | "
            f"{draft_top1:.4f} | {draft_top5:.4f} | {row['correction_rate']:.4f} | "
            f"{row['regression_rate']:.4f} | {row['net_correction']:.4f} | "
            f"{row['selected_token_ratio']:.4f} | {block_em:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"1. AR draft 是否造成了 MDLM 的 draft-context 性能下降？{'是' if draft_drop else '否或证据不足'}。",
            f"2. LoRA 微调是否提升了 MDLM 在 GPT-2 draft context 下的恢复能力？{'是' if lora_beats_draft else '否或证据不足'}。",
            f"3. LoRA-MDLM 是否比 pretrained MDLM 更适合修正 GPT-2 draft？{'是' if lora_beats_refine else '否或证据不足'}。",
            f"4. correction_rate 是否提升？{'是' if corr_up else '否或证据不足'}。",
            f"5. regression_rate 是否下降？{'是' if reg_down else '否或证据不足'}。",
            f"6. net_correction 是否接近 0 或转正？{'是' if net_good else '否'}。",
            "7. block-level draft-aware training 是否优于 token-level refinement？当前脚本只实现 block-level 训练；需要另跑 token-level ablation 才能严格回答。",
            f"8. 当前结果是否支持“AR draft 可以作为扩散语言模型训练噪声源”的研究假设？{'支持' if hypothesis else '暂不充分支持'}。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str, lora_path: str | None, eval_steps: int | None = None, save_dir: str | None = None) -> list[dict]:
    config = load_config(config_path)
    config["mdlm_ckpt"] = None
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)
    if save_dir is not None:
        config["save_dir"] = save_dir
    rows = evaluate(config, lora_path)
    out_dir = Path(config["save_dir"])
    save_csv(out_dir / "eval_metrics.csv", rows)
    save_json(out_dir / "eval_metrics.json", {"benchmark": rows})
    write_summary(out_dir / "draft_aware_lora_summary.md", rows, config)
    display = [{key: row[key] for key in EVAL_COLUMNS if key in row} for row in rows]
    print(format_table(display, EVAL_COLUMNS))
    print(f"saved_eval_metrics={out_dir / 'eval_metrics.csv'}")
    print(f"saved_summary={out_dir / 'draft_aware_lora_summary.md'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    run(args.config, args.lora_path, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
