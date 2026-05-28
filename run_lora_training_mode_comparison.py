"""Compare pretrained, random-mask LoRA, and draft-aware LoRA MDLM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eval_draft_aware_lora import evaluate
from refine_utils import load_config


COLUMNS = [
    "Model",
    "Train Noise",
    "Standard Top1",
    "Standard Top5",
    "Draft Top1",
    "Draft Top5",
    "Draft PPL",
    "Correction",
    "Regression",
    "Net",
]


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_from_eval(label: str, train_noise: str, rows: list[dict], prefix: str) -> dict:
    by_mode = {row["mode"]: row for row in rows}
    standard = by_mode[f"{prefix}_mdlm_standard"]
    draft = by_mode[f"{prefix}_mdlm_draft_context"]
    refine = by_mode[f"gpt2_plus_{prefix}_mdlm_refine"]
    return {
        "Model": label,
        "Train Noise": train_noise,
        "Standard Top1": standard["top1"],
        "Standard Top5": standard["top5"],
        "Draft Top1": draft["draft_context_top1"],
        "Draft Top5": draft["draft_context_top5"],
        "Draft PPL": draft["ppl"],
        "Correction": refine["correction_rate"],
        "Regression": refine["regression_rate"],
        "Net": refine["net_correction"],
    }


def pretrained_row(rows: list[dict]) -> dict:
    by_mode = {row["mode"]: row for row in rows}
    standard = by_mode["pretrained_mdlm_standard"]
    draft = by_mode["pretrained_mdlm_draft_context"]
    refine = by_mode["gpt2_plus_pretrained_mdlm_refine"]
    return {
        "Model": "pretrained_mdlm",
        "Train Noise": "pretrained",
        "Standard Top1": standard["top1"],
        "Standard Top5": standard["top5"],
        "Draft Top1": draft["draft_context_top1"],
        "Draft Top5": draft["draft_context_top5"],
        "Draft PPL": draft["ppl"],
        "Correction": refine["correction_rate"],
        "Regression": refine["regression_rate"],
        "Net": refine["net_correction"],
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    by_noise = {row["Train Noise"]: row for row in rows}
    random_row = by_noise.get("random_mask")
    draft_row = by_noise.get("draft_aware")
    better_draft = bool(random_row and draft_row and draft_row["Draft Top1"] > random_row["Draft Top1"])
    better_net = bool(random_row and draft_row and draft_row["Net"] > random_row["Net"])
    lines = [
        "# LoRA Training Mode Comparison",
        "",
        "| Model | Train Noise | Standard Top1 | Standard Top5 | Draft Top1 | Draft Top5 | Draft PPL | Correction | Regression | Net |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Train Noise']} | {row['Standard Top1']:.4f} | "
            f"{row['Standard Top5']:.4f} | {row['Draft Top1']:.4f} | {row['Draft Top5']:.4f} | "
            f"{row['Draft PPL']:.4f} | {row['Correction']:.4f} | {row['Regression']:.4f} | {row['Net']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answer",
            "",
            f"1. draft-aware LoRA 在 GPT-2 draft context 下是否优于 random-mask LoRA？{'是' if better_draft else '否或证据不足'}。",
            f"2. draft-aware LoRA 的 refinement net correction 是否优于 random-mask LoRA？{'是' if better_net else '否或证据不足'}。",
            f"3. 该结果是否支持 AR draft 作为结构化训练噪声源？{'支持' if better_draft else '暂不充分支持'}。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    eval_config_path: str,
    random_lora_path: str,
    draft_lora_path: str,
    eval_steps: int | None,
    save_dir: str,
) -> list[dict]:
    config = load_config(eval_config_path)
    config["mdlm_ckpt"] = None
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)

    pretrained_rows = evaluate(config, None)
    random_rows = evaluate(config, random_lora_path)
    draft_rows = evaluate(config, draft_lora_path)
    rows = [
        pretrained_row(pretrained_rows),
        row_from_eval("random_mask_lora_mdlm", "random_mask", random_rows, "lora"),
        row_from_eval("draft_aware_lora_mdlm", "draft_aware", draft_rows, "lora"),
    ]

    out_dir = Path(save_dir)
    save_csv(out_dir / "training_mode_comparison.csv", rows)
    save_json(
        out_dir / "training_mode_comparison.json",
        {"benchmark": rows, "eval_config": eval_config_path, "random_lora_path": random_lora_path, "draft_lora_path": draft_lora_path},
    )
    write_summary(out_dir / "training_mode_comparison_summary.md", rows)
    print(f"saved_training_mode_comparison={out_dir / 'training_mode_comparison.csv'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config", default="configs/wikitext2_draft_aware_lora.yaml")
    parser.add_argument("--random_lora_path", default="results/wikitext2_random_mask_lora/lora_adapter")
    parser.add_argument("--draft_lora_path", default="results/wikitext2_draft_aware_lora/lora_adapter")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default="results/lora_training_mode_comparison")
    args = parser.parse_args()
    run(args.eval_config, args.random_lora_path, args.draft_lora_path, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
