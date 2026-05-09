"""Print and refresh the saved coarse-to-fine comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from metrics import format_table, load_results, write_results


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved coarse-to-fine experiment results.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = config["outputs"]["dir"]
    results = load_results(output_dir)
    write_results(results, output_dir)
    expected = {"device_only", "edge_only", "vanilla_collaborative_distillation", "coarse_to_fine"}
    missing = sorted(expected - {row.get("mode") for row in results})
    if missing:
        print(f"missing modes: {', '.join(missing)}")
    print(format_table(results))
    print(f"saved: {Path(output_dir) / 'coarse_to_fine_results.json'}")
    print(f"saved: {Path(output_dir) / 'coarse_to_fine_results.csv'}")


if __name__ == "__main__":
    main()
