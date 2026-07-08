"""Train and evaluate draft-aware LoRA adapters for multiple block sizes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from eval_final_refinement import evaluate
from draft_utils import load_config
from train_draft_aware_lora import train


COLUMNS = [
    "Block Size",
    "Draft Top1",
    "Draft Top5",
    "PPL",
    "Correction",
    "Regression",
    "Net",
    "Selected Ratio",
    "latency",
    "tokens_per_sec",
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


def write_summary(path: Path, rows: list[dict]) -> None:
    by_block = {int(row["Block Size"]): row for row in rows}
    token = by_block.get(1)
    best = max(rows, key=lambda row: (row["Draft Top1"], row["Net"])) if rows else None
    block4 = by_block.get(4)
    largest = by_block.get(max(by_block)) if by_block else None
    block4_beats_token = bool(token and block4 and block4["Draft Top1"] > token["Draft Top1"])
    block_training_helps = bool(token and best and best["Draft Top1"] >= token["Draft Top1"] and best["Net"] >= token["Net"])
    too_large_hurts = bool(best and largest and largest is not best and largest["Draft Top1"] < best["Draft Top1"])

    lines = [
        "# Block-size Ablation",
        "",
        "| Block Size | Draft Top1 | Draft Top5 | PPL | Correction | Regression | Net | Selected Ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['Block Size'])} | {row['Draft Top1']:.4f} | {row['Draft Top5']:.4f} | "
            f"{row['PPL']:.4f} | {row['Correction']:.4f} | "
            f"{row['Regression']:.4f} | {row['Net']:.4f} | {row['Selected Ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. block_size=1 当前效果最好：{'是' if best and int(best['Block Size']) == 1 else '否或证据不足'}。",
            f"2. 固定大 block 不适合作为主配置：{'是' if not block4_beats_token or too_large_hurts else '否或证据不足'}。",
            "3. 当前最终方法默认使用 token-level draft-aware LoRA。",
            f"4. 当前最优 block_size 是多少？`{int(best['Block Size']) if best else ''}`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def summarize_eval(block_size: int, rows: list[dict]) -> dict:
    by_mode = {row["mode"]: row for row in rows}
    refine = by_mode["draft_aware_lora_refine_no_gate"]
    return {
        "Block Size": int(block_size),
        "Draft Top1": refine["top1"],
        "Draft Top5": refine["top5"],
        "PPL": refine["ppl"],
        "Correction": refine["correction_rate"],
        "Regression": refine["regression_rate"],
        "Net": refine["net_correction"],
        "Selected Ratio": refine["selected_token_ratio"],
        "latency": refine["latency"],
        "tokens_per_sec": refine["tokens_per_sec"],
    }


def run(base_config: str, block_sizes: list[int], train_steps: int | None, eval_steps: int | None, save_dir: str) -> list[dict]:
    base = load_config(base_config)
    base["lora_training_mode"] = "draft_aware"
    base["mdlm_ckpt"] = None
    out_dir = Path(save_dir)
    rows: list[dict] = []
    for block_size in block_sizes:
        run_dir = out_dir / f"block_size_{int(block_size)}"
        config = dict(base)
        config["block_size"] = int(block_size)
        config["save_dir"] = str(run_dir)
        config["draft_aware_lora_path"] = str(run_dir / "lora_adapter")
        config["use_accept_gate"] = False
        if train_steps is not None:
            config["train_steps"] = int(train_steps)
        if eval_steps is not None:
            config["eval_steps"] = int(eval_steps)
            config["eval_batches"] = max(1, min(int(eval_steps), int(config.get("eval_batches", eval_steps))))
        config_path = out_dir / "configs" / f"block_size_{int(block_size)}.yaml"
        write_config(config_path, config)
        print(f"training_block_size={block_size} config={config_path}", flush=True)
        train(str(config_path))
        eval_rows = evaluate(config, None)
        rows.append(summarize_eval(int(block_size), eval_rows))
        save_csv(out_dir / "block_size_ablation.csv", rows)
        save_json(out_dir / "block_size_ablation.json", {"benchmark": rows})
        write_summary(out_dir / "block_size_ablation_summary.md", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--block_sizes", nargs="+", type=int, required=True)
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default="results/block_size_ablation")
    args = parser.parse_args()
    run(args.base_config, args.block_sizes, args.train_steps, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
