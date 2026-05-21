"""Run coarse-dimension ablation for the MDLM coarse-to-fine prototype."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import yaml


ABLATION_COLUMNS = [
    "coarse_dim",
    "communication_MB",
    "edge_only_ppl",
    "coarse_to_fine_ppl",
    "top1_gain",
    "top5_gain",
    "quality_gain_per_MB",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def summarize(save_dir: Path, coarse_dim: int) -> dict:
    rows = json.loads((save_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    edge = next(row for row in rows if row["mode"] == "edge_only")
    ctf = next(row for row in rows if row["mode"] == "coarse_to_fine")
    return {
        "coarse_dim": coarse_dim,
        "communication_MB": ctf["communication_MB"],
        "edge_only_ppl": edge["perplexity"],
        "coarse_to_fine_ppl": ctf["perplexity"],
        "top1_gain": ctf["top1_acc"] - edge["top1_acc"],
        "top5_gain": ctf["top5_acc"] - edge["top5_acc"],
        "quality_gain_per_MB": ctf["quality_gain_per_MB"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ABLATION_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--coarse_dims", nargs="+", type=int, required=True)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base_config = load_yaml(base_path)
    base_save_dir = Path(base_config["save_dir"])
    rows = []

    for coarse_dim in args.coarse_dims:
        config = dict(base_config)
        config["coarse_dim"] = coarse_dim
        config["save_dir"] = str(base_save_dir.with_name(f"{base_save_dir.name}_coarse{coarse_dim}"))
        config_path = base_save_dir.parent / f"{base_path.stem}_coarse{coarse_dim}.yaml"
        write_yaml(config_path, config)
        ckpt_path = Path(config["save_dir"]) / "checkpoint.pt"

        if not args.skip_train:
            run_command(["python", "train_coarse_to_fine.py", "--config", str(config_path)])
        run_command(["python", "eval_coarse_to_fine.py", "--config", str(config_path), "--ckpt", str(ckpt_path)])
        rows.append(summarize(Path(config["save_dir"]), coarse_dim))

    output_csv = base_save_dir.parent / "coarse_dim_ablation.csv"
    output_json = base_save_dir.parent / "coarse_dim_ablation.json"
    write_csv(output_csv, rows)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved_ablation_csv={output_csv}")
    print(f"saved_ablation_json={output_json}")


if __name__ == "__main__":
    main()
