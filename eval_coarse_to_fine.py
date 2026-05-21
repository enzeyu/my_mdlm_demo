"""Evaluate device-only, edge-only, and coarse-to-fine denoising quality."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import torch
import yaml

from data_real import build_dataloaders, mask_tokens
from metrics import format_table, gpu_memory_mb, now, reset_gpu_memory, sync_if_cuda, write_csv, write_json
from model_coarse_to_fine import build_model_from_config, coarse_comm_mb, compression_ratio, masked_lm_metrics

BENCHMARK_COLUMNS = [
    "mode",
    "model_backend",
    "loss",
    "perplexity",
    "top1_acc",
    "top5_acc",
    "latency",
    "tokens_per_sec",
    "gpu_memory_MB",
    "communication_MB",
    "compression_ratio",
    "gain_over_edge_only",
    "quality_gain_per_MB",
]


def load_config(path: str) -> dict:
    """Load evaluation YAML config."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device() -> torch.device:
    """Use CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(config: dict, ckpt_arg: str | None) -> Path:
    """Use explicit checkpoint path or the default checkpoint in save_dir."""
    if ckpt_arg:
        return Path(ckpt_arg)
    return Path(config["save_dir"]) / "checkpoint.pt"


def communication_mb_for_mode(config: dict, mode: str) -> float:
    """Estimate transferred coarse-state MiB for one evaluated batch."""
    if mode != "coarse_to_fine":
        return 0.0
    dtype_bytes = 2 if config.get("precision", "fp32") in {"fp16", "bf16"} else 4
    return coarse_comm_mb(
        int(config["batch_size"]),
        int(config["max_length"]),
        int(config["coarse_dim"]),
        dtype_bytes=dtype_bytes,
    )


@torch.no_grad()
def evaluate_mode(model, val_loader, config, tokenizer_info, device, mode: str):
    """Evaluate one mode and return LM quality, latency, and systems metrics."""
    model.eval()
    #reset_gpu_memory(device)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_tokens = 0
    total_latency = 0.0
    batches = 0

    for batch in itertools.islice(val_loader, int(config["eval_steps"])):
        clean = batch.to(device, non_blocking=True)
        noised, target_mask = mask_tokens(clean, tokenizer_info.mask_token_id, tokenizer_info.pad_token_id, float(config["mask_ratio"]))
        timesteps = torch.full((clean.size(0),), float(config["mask_ratio"]), device=device)

        sync_if_cuda(device)
        start = now()
        outputs = model(noised, timesteps, mode=mode)
        metrics = masked_lm_metrics(outputs["logits"], clean, target_mask)
        sync_if_cuda(device)

        latency = now() - start
        num_tokens = int(metrics["num_tokens"])
        total_latency += latency
        total_loss += float(metrics["loss"].item()) * num_tokens
        total_top1 += float(metrics["top1_acc"].item()) * num_tokens
        total_top5 += float(metrics["top5_acc"].item()) * num_tokens
        total_tokens += num_tokens
        batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    avg_latency = total_latency / max(batches, 1)
    comm_mb = communication_mb_for_mode(config, mode)
    edge_hidden_size = int(getattr(model, "edge_hidden_size", config.get("edge_hidden_size", 384) if config.get("edge_hidden_size") != "auto" else 768))
    return {
        "mode": mode,
        "model_backend": getattr(model, "backend_name", config.get("model_backend", "internal_toy")),
        "loss": avg_loss,
        "perplexity": float(math.exp(min(avg_loss, 50.0))),
        "top1_acc": total_top1 / max(total_tokens, 1),
        "top5_acc": total_top5 / max(total_tokens, 1),
        "latency": avg_latency,
        "tokens_per_sec": total_tokens / max(total_latency, 1e-9),
        "gpu_memory_MB": gpu_memory_mb(device),
        "communication_MB": comm_mb,
        "compression_ratio": compression_ratio(edge_hidden_size, int(config["coarse_dim"])),
        "gain_over_edge_only": 0.0,
        "quality_gain_per_MB": 0.0,
    }


def save_benchmark_csv(path: Path, rows: list[dict]) -> None:
    """Save benchmark rows with the exact research-facing column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in BENCHMARK_COLUMNS})


def load_training_rows(save_dir: Path) -> list[dict]:
    """Load prior training metrics when available for curve visualizations."""
    jsonl_path = save_dir / "train_metrics.jsonl"
    if not jsonl_path.exists():
        return []
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(yaml.safe_load(line))
            except yaml.YAMLError:
                continue
    return rows


def metric_series(rows: list[dict], *names: str) -> tuple[list[int], list[float]]:
    """Extract a training metric series using the first available key name."""
    steps = []
    values = []
    for index, row in enumerate(rows, start=1):
        key = next((name for name in names if row.get(name) not in (None, "")), None)
        if key is None:
            continue
        steps.append(int(row.get("step", index)))
        values.append(float(row[key]))
    return steps, values


def plot_curve(save_dir: Path, filename: str, ylabel: str, train_rows: list[dict], benchmark_rows: list[dict], series_specs):
    """Plot training/validation curves, falling back to benchmark mode comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for label, names in series_specs:
        xs, ys = metric_series(train_rows, *names)
        if xs and ys:
            ax.plot(xs, ys, label=label, linewidth=1.8)
            plotted = True

    if not plotted:
        candidate_names = [name for _, names in series_specs for name in names]
        metric_name = next((name for name in candidate_names if name in benchmark_rows[0]), candidate_names[0])
        modes = [row["mode"] for row in benchmark_rows]
        values = [float(row[metric_name]) for row in benchmark_rows]
        ax.plot(range(len(modes)), values, marker="o", linewidth=1.8)
        ax.set_xticks(range(len(modes)), modes, rotation=15)
        ax.set_xlabel("mode")
    else:
        ax.set_xlabel("step")
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_dir / filename, dpi=160)
    plt.close(fig)


def plot_quality_vs_comm(save_dir: Path, benchmark_rows: list[dict]) -> None:
    """Plot top-5 denoising quality against communication overhead."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for row in benchmark_rows:
        x = float(row["communication_MB"])
        y = float(row["top5_acc"])
        ax.scatter([x], [y], s=70)
        ax.annotate(row["mode"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("communication_MB")
    ax.set_ylabel("top5_acc")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_dir / "quality_vs_comm.png", dpi=160)
    plt.close(fig)


def save_visualizations(save_dir: Path, benchmark_rows: list[dict]) -> None:
    """Generate benchmark and training metric figures."""
    train_rows = load_training_rows(save_dir)
    plot_curve(
        save_dir,
        "ppl_curve.png",
        "perplexity",
        train_rows,
        benchmark_rows,
        [("train", ("train_perplexity", "perplexity")), ("val", ("val_perplexity",))],
    )
    plot_curve(
        save_dir,
        "top1_acc_curve.png",
        "top1_acc",
        train_rows,
        benchmark_rows,
        [("train", ("top1_acc", "token_acc")), ("val", ("val_top1_acc", "val_token_acc"))],
    )
    plot_curve(
        save_dir,
        "top5_acc_curve.png",
        "top5_acc",
        train_rows,
        benchmark_rows,
        [("train", ("top5_acc", "train_top5_acc")), ("val", ("val_top5_acc",))],
    )
    plot_quality_vs_comm(save_dir, benchmark_rows)


def main() -> None:
    """Load checkpoint, evaluate three modes, and save result tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    model = build_model_from_config(
        config,
        tokenizer_info.vocab_size,
        tokenizer_info.pad_token_id,
        tokenizer_info.mask_token_id,
    ).to(device)
    print(f"model_backend={getattr(model, 'backend_name', config.get('model_backend', 'internal_toy'))}")
    print(f"pretrained_edge_loaded={getattr(model, 'pretrained_loaded', False)}")
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")
    print(f"edge_model_is_toy={getattr(model, 'backend_name', 'internal_toy') == 'internal_toy'}")

    ckpt_path = resolve_checkpoint(config, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(f"checkpoint_load_warning missing={len(missing)} unexpected={len(unexpected)}")

    rows = []
    for mode in ["device_only", "edge_only", "coarse_to_fine"]:
        rows.append(evaluate_mode(model, val_loader, config, tokenizer_info, device, mode))

    device_row = next(row for row in rows if row["mode"] == "device_only")
    edge_row = next(row for row in rows if row["mode"] == "edge_only")
    ctf_row = next(row for row in rows if row["mode"] == "coarse_to_fine")
    refinement_gain = ctf_row["top1_acc"] - edge_row["top1_acc"]
    refinement_top5_gain = ctf_row["top5_acc"] - edge_row["top5_acc"]
    ppl_improvement = edge_row["perplexity"] - ctf_row["perplexity"]
    communication_mb = ctf_row["communication_MB"]
    quality_gain_per_mb = refinement_gain / max(communication_mb, 1e-12)
    top5_quality_gain_per_mb = refinement_top5_gain / max(communication_mb, 1e-12)
    for row in rows:
        row["gain_over_device_top1"] = row["top1_acc"] - device_row["top1_acc"]
        row["gain_over_edge_top1"] = row["top1_acc"] - edge_row["top1_acc"]
        row["gain_over_edge_top5"] = row["top5_acc"] - edge_row["top5_acc"]
        row["gain_over_edge_only"] = row["top1_acc"] - edge_row["top1_acc"]
        row["quality_gain_per_MB"] = row["gain_over_edge_only"] / max(float(row["communication_MB"]), 1e-12)

    save_dir = Path(config["save_dir"])
    summary = {
        "refinement_gain": refinement_gain,
        "refinement_top5_gain": refinement_top5_gain,
        "coarse_to_fine_ppl_improvement": ppl_improvement,
        "communication_MB": communication_mb,
        "compression_ratio": ctf_row["compression_ratio"],
        "quality_gain_per_MB": quality_gain_per_mb,
        "top5_quality_gain_per_MB": top5_quality_gain_per_mb,
        "coarse_to_fine_beats_edge_only_top1": ctf_row["top1_acc"] > edge_row["top1_acc"],
        "coarse_to_fine_beats_edge_only_top5": ctf_row["top5_acc"] > edge_row["top5_acc"],
        "coarse_to_fine_beats_edge_only_ppl": ctf_row["perplexity"] < edge_row["perplexity"],
    }
    write_json(save_dir / "eval_metrics.json", [{"benchmark": rows, "summary": summary}])
    write_json(save_dir / "benchmark_results.json", rows)
    write_csv(save_dir / "eval_metrics.csv", rows)
    save_benchmark_csv(save_dir / "benchmark_results.csv", rows)
    summary_lines = [
        f"model_backend: {getattr(model, 'backend_name', config.get('model_backend', 'internal_toy'))}",
        f"pretrained_edge_loaded: {getattr(model, 'pretrained_loaded', False)}",
        f"coarse_to_fine_better_than_edge_only_top1: {summary['coarse_to_fine_beats_edge_only_top1']}",
        f"perplexity_decreased: {summary['coarse_to_fine_beats_edge_only_ppl']}",
        f"top1_gain: {refinement_gain:.6f}",
        f"top5_gain: {refinement_top5_gain:.6f}",
        f"communication_MB: {communication_mb:.6f}",
        f"quality_gain_per_MB: {quality_gain_per_mb:.6f}",
        f"latency_delta_vs_edge_only: {ctf_row['latency'] - edge_row['latency']:.6f}",
        "hypothesis_supported: "
        + str(
            summary["coarse_to_fine_beats_edge_only_ppl"]
            or summary["coarse_to_fine_beats_edge_only_top1"]
            or summary["coarse_to_fine_beats_edge_only_top5"]
        ),
    ]
    (save_dir / "benchmark_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    save_visualizations(save_dir, rows)

    print(format_table(rows, ["mode", "perplexity", "top1_acc", "top5_acc"]))
    print()
    print(format_table(rows, ["mode", "loss", "latency", "tokens_per_sec", "gpu_memory_MB", "communication_MB", "compression_ratio"]))
    print()
    print(format_table([summary], [
        "refinement_gain",
        "refinement_top5_gain",
        "coarse_to_fine_ppl_improvement",
        "communication_MB",
        "quality_gain_per_MB",
        "top5_quality_gain_per_MB",
    ]))
    print(f"saved_benchmark={save_dir / 'benchmark_results.csv'}")
    print(f"saved_visualizations={save_dir / 'ppl_curve.png'}, {save_dir / 'top1_acc_curve.png'}, {save_dir / 'top5_acc_curve.png'}, {save_dir / 'quality_vs_comm.png'}")


if __name__ == "__main__":
    main()
