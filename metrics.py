"""Metric aggregation and result persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def write_results(results: List[Dict[str, float]], output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "results.json"
    csv_path = path / "results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fieldnames = [
        "mode",
        "train_time",
        "step_time",
        "memory_MB",
        "comm_MB",
        "sampling_latency",
        "tokens_per_sec",
        "token_acc",
        "val_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_results(output_dir: str) -> List[Dict[str, float]]:
    path = Path(output_dir) / "results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_result(results: Iterable[Dict[str, float]], new_result: Dict[str, float]) -> List[Dict[str, float]]:
    rows = [row for row in results if row.get("mode") != new_result.get("mode")]
    rows.append(new_result)
    order = {"device_only": 0, "edge_only": 1, "collaborative": 2}
    return sorted(rows, key=lambda row: order.get(str(row.get("mode")), 99))


def format_table(results: Iterable[Dict[str, float]]) -> str:
    columns = [
        "mode",
        "train_time",
        "step_time",
        "memory_MB",
        "comm_MB",
        "sampling_latency",
        "tokens_per_sec",
        "token_acc",
        "val_loss",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                str(result.get("mode", "")),
                f"{float(result.get('train_time', 0.0)):.3f}",
                f"{float(result.get('step_time', 0.0)):.4f}",
                f"{float(result.get('memory_MB', 0.0)):.2f}",
                f"{float(result.get('comm_MB', 0.0)):.4f}",
                f"{float(result.get('sampling_latency', 0.0)):.4f}",
                f"{float(result.get('tokens_per_sec', 0.0)):.2f}",
                f"{float(result.get('token_acc', 0.0)):.4f}",
                f"{float(result.get('val_loss', 0.0)):.4f}",
            ]
        )
    widths = [len(col) for col in columns]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = " | ".join(col.ljust(width) for col, width in zip(columns, widths))
    sep = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows]
    return "\n".join([header, sep, *body])

