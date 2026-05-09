"""CLI entry point for one coarse-to-fine experiment mode."""

from __future__ import annotations

import argparse

import yaml

from metrics import format_table, load_results
from trainer import ExperimentTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a coarse-to-fine edge-device diffusion LM demo.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["device_only", "edge_only", "vanilla_collaborative_distillation", "coarse_to_fine", "collaborative"],
        required=True,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = ExperimentTrainer(config, args.mode).train()
    print(f"finished mode={result['mode']}")
    print(format_table(load_results(config["outputs"]["dir"])))
    print(f"latest token_acc={result['token_acc']:.4f} val_loss={result['val_loss']:.4f}")


if __name__ == "__main__":
    main()
