"""Metric aggregation and result persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


RESULT_FIELDS = [
    "mode",
    "dataset",
    "coarse_dim",
    "compression_method",
    "device_steps",
    "edge_steps",
    "train_time",
    "step_time",
    "memory_MB",
    "comm_MB",
    "sampling_latency",
    "tokens_per_sec",
    "token_acc",
    "val_loss",
    "compression_ratio",
    "refinement_gain",
]


def write_results(results: List[Dict[str, float]], output_dir: str, stem: str = "coarse_to_fine_results") -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / f"{stem}.json"
    csv_path = path / f"{stem}.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})


def load_results(output_dir: str, stem: str = "coarse_to_fine_results") -> List[Dict[str, float]]:
    path = Path(output_dir) / f"{stem}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_result(results: Iterable[Dict[str, float]], new_result: Dict[str, float]) -> List[Dict[str, float]]:
    key = (
        new_result.get("mode"),
        new_result.get("coarse_dim"),
        new_result.get("compression_method"),
        new_result.get("device_steps"),
        new_result.get("edge_steps"),
    )
    rows = []
    for row in results:
        row_key = (row.get("mode"), row.get("coarse_dim"), row.get("compression_method"), row.get("device_steps"), row.get("edge_steps"))
        if row_key != key:
            rows.append(row)
    rows.append(new_result)
    return sorted(rows, key=lambda r: (str(r.get("mode")), int(r.get("coarse_dim", 0)), str(r.get("compression_method"))))


def format_table(results: Iterable[Dict[str, float]]) -> str:
    columns = ["mode", "coarse_dim", "compression_method", "comm_MB", "sampling_latency", "tokens_per_sec", "token_acc", "val_loss", "refinement_gain"]
    rows = []
    for r in results:
        rows.append(
            [
                str(r.get("mode", "")),
                str(r.get("coarse_dim", "")),
                str(r.get("compression_method", "")),
                f"{float(r.get('comm_MB', 0.0)):.4f}",
                f"{float(r.get('sampling_latency', 0.0)):.4f}",
                f"{float(r.get('tokens_per_sec', 0.0)):.2f}",
                f"{float(r.get('token_acc', 0.0)):.4f}",
                f"{float(r.get('val_loss', 0.0)):.4f}",
                f"{float(r.get('refinement_gain', 0.0)):.4f}",
            ]
        )
    widths = [len(c) for c in columns]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = [" | ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows]
    return "\n".join([header, sep, *body]) if body else header
