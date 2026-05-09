"""CLI entry point for training one experimental mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_device_training.metrics import format_table, load_results
from edge_device_training.trainer import ExperimentTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an edge-device diffusion LM prototype.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["device_only", "edge_only", "collaborative"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = ExperimentTrainer(config, args.mode)
    result = trainer.train()
    print(f"finished mode={args.mode}")
    print(format_table(load_results(config["outputs"]["dir"])))
    print(f"latest train_loss={result['train_loss']:.4f} val_loss={result['val_loss']:.4f}")


if __name__ == "__main__":
    main()

