"""Metric helpers for the real PyTorch coarse-to-fine prototype."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def now() -> float:
    """Return a high-resolution timestamp for latency measurement."""
    return time.perf_counter()


def gpu_memory_mb(device: torch.device) -> float:
    """Return peak allocated CUDA memory in MiB, or zero on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def reset_gpu_memory(device: torch.device) -> None:
    """Reset CUDA peak memory stats before a measured section."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA so wall-clock timings include queued GPU kernels."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def append_jsonl(path: str | Path, row: Dict) -> None:
    """Append one metric row to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: str | Path, rows: List[Dict]) -> None:
    """Write a list of metric rows as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[Dict]) -> None:
    """Write metric rows to CSV using the union of all observed keys."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_table(rows: Iterable[Dict], columns: List[str]) -> str:
    """Format selected metric columns as a readable terminal table."""
    rows = list(rows)
    formatted = []
    for row in rows:
        formatted.append([_fmt(row.get(col, "")) for col in columns])
    widths = [len(col) for col in columns]
    for row in formatted:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = " | ".join(col.ljust(width) for col, width in zip(columns, widths))
    sep = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in formatted]
    return "\n".join([header, sep, *body])


def _fmt(value) -> str:
    """Format floats compactly while leaving strings unchanged."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
