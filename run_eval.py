"""Print the baseline comparison table saved by training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_device_training.metrics import format_table, load_results, write_results


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved edge-device training results.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = config["outputs"]["dir"]
    results = load_results(output_dir)
    order = {"device_only": 0, "edge_only": 1, "collaborative": 2}
    results = sorted(results, key=lambda row: order.get(str(row.get("mode")), 99))
    write_results(results, output_dir)
    missing = [mode for mode in order if mode not in {row.get("mode") for row in results}]
    if missing:
        print(f"missing modes: {', '.join(missing)}")
        print("run run_train.py for each missing mode to complete the comparison.")
    print(format_table(results))
    print(f"saved: {Path(output_dir) / 'results.json'}")
    print(f"saved: {Path(output_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()

