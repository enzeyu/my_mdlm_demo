"""Run baseline and coarse-space ablation experiments."""

from __future__ import annotations

import argparse
import copy

import yaml

from metrics import format_table, load_results, upsert_result, write_results
from trainer import ExperimentTrainer, with_overrides


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coarse-to-fine ablations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    output_dir = base["outputs"]["dir"]
    results = load_results(output_dir)

    baseline_modes = ["device_only", "edge_only", "vanilla_collaborative_distillation", "coarse_to_fine"]
    for mode in baseline_modes:
        cfg = copy.deepcopy(base)
        print(f"running baseline mode={mode}")
        row = ExperimentTrainer(cfg, mode).train(persist=False)
        results = upsert_result(results, row)
        write_results(results, output_dir)

    max_ablation_runs = int(base.get("ablation", {}).get("max_runs", 8))
    coarse_dims = base.get("ablation", {}).get("coarse_dims", [32, 64, 128])
    methods = base.get("ablation", {}).get("compression_methods", ["linear", "pooling"])
    device_steps = base.get("ablation", {}).get("device_steps", [2, 4, 8])
    edge_steps = base.get("ablation", {}).get("edge_steps", [1, 2, 4])

    runs = 0
    for dim in coarse_dims:
        for method in methods:
            for ds in device_steps:
                for es in edge_steps:
                    if runs >= max_ablation_runs:
                        break
                    cfg = with_overrides(base, coarse_dim=dim, compression_method=method, device_steps=ds, edge_steps=es)
                    print(f"running ablation dim={dim} method={method} device_steps={ds} edge_steps={es}")
                    row = ExperimentTrainer(cfg, "coarse_to_fine").train(persist=False)
                    results = upsert_result(results, row)
                    write_results(results, output_dir)
                    runs += 1
                if runs >= max_ablation_runs:
                    break
            if runs >= max_ablation_runs:
                break
        if runs >= max_ablation_runs:
            break

    by_key = {(r.get("coarse_dim"), r.get("compression_method"), r.get("device_steps"), r.get("edge_steps")): r for r in results}
    for row in results:
        if row.get("mode") == "coarse_to_fine":
            key = (row.get("coarse_dim"), row.get("compression_method"), row.get("device_steps"), row.get("edge_steps"))
            device_row = next((r for r in results if r.get("mode") == "device_only"), None)
            if device_row:
                row["refinement_gain"] = float(row.get("token_acc", 0.0)) - float(device_row.get("token_acc", 0.0))
            by_key[key] = row
    write_results(results, output_dir)
    print(format_table(results))


if __name__ == "__main__":
    main()
