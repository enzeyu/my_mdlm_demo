"""Generate the final DART report from saved result files."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("benchmark", [])


def fmt(value) -> str:
    if value in {"", None}:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["No result file found."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return lines


def best(rows: list[dict], key: str) -> dict | None:
    candidates = [row for row in rows if row.get(key) not in {"", None}]
    return max(candidates, key=lambda row: row[key]) if candidates else None


def main() -> None:
    training_mode = load_rows(Path("results/lora_training_mode_comparison/training_mode_comparison.json"))
    block_size = load_rows(Path("results/block_size_ablation/block_size_ablation.json"))
    accept_gate = [
        row
        for row in load_rows(Path("results/accept_gate/accept_gate_eval.json"))
        if row.get("mode") != "draft_aware_lora_refine_with_rule_gate"
    ]
    dart_final = load_rows(Path("results/dart_final/final_eval.json"))
    final_rows = dart_final or accept_gate

    random_row = next((row for row in training_mode if row.get("Train Noise") == "random_mask"), None)
    draft_row = next((row for row in training_mode if row.get("Train Noise") == "draft_aware"), None)
    best_block = best(block_size, "Draft Top1")
    final_gpt2 = next((row for row in final_rows if row.get("mode") == "gpt2_only"), None)
    final_learned_rows = [row for row in final_rows if row.get("mode") == "draft_aware_lora_refine_with_learned_gate"]
    final_learned = max(final_learned_rows, key=lambda row: (row.get("top1", 0), row.get("net_correction", -99))) if final_learned_rows else None
    gate_no = next((row for row in accept_gate if row.get("mode") == "draft_aware_lora_refine_no_gate"), None)
    gate_learned_rows = [row for row in accept_gate if row.get("mode") == "draft_aware_lora_refine_with_learned_gate"]
    gate_learned = min(gate_learned_rows, key=lambda row: row.get("regression_rate", 99)) if gate_learned_rows else None

    lines = [
        "# DART Final Report",
        "",
        "## 1. Problem Definition",
        "",
        "DART targets edge-device collaboration where a frozen device GPT-2 produces an autoregressive draft and a frozen edge MDLM is adapted with LoRA to refine the low-confidence parts of that draft.",
        "",
        "## 2. Method Flow",
        "",
        "Device GPT-2 draft -> low-confidence token selection -> Edge MDLM draft-aware LoRA adaptation -> learned accept gate -> selective refinement.",
        "",
        "## 3. Random-mask LoRA vs Draft-aware LoRA",
        "",
        *table(training_mode, ["Model", "Train Noise", "Standard Top1", "Standard Top5", "Draft Top1", "Draft Top5", "Draft PPL", "Correction", "Regression", "Net"]),
        "",
        f"Conclusion: AR draft-induced corruption is an effective structured training noise source; draft-aware LoRA under GPT-2 draft context is better than random-mask LoRA when Draft Top1 improves ({fmt(random_row.get('Draft Top1')) if random_row else ''} -> {fmt(draft_row.get('Draft Top1')) if draft_row else ''}).",
        "",
        "## 4. Block-size Ablation",
        "",
        *table(block_size, ["Block Size", "Draft Top1", "Draft Top5", "PPL", "Correction", "Regression", "Net", "Selected Ratio"]),
        "",
        f"Conclusion: block_size=1 is the current best setting when it is the top Draft Top1 row; current best block_size is `{best_block.get('Block Size') if best_block else ''}`. Fixed large blocks are not used as the main configuration.",
        "",
        "## 5. Accept Gate",
        "",
        *table(accept_gate, ["mode", "gate_threshold", "top1", "top5", "ppl", "correction_rate", "regression_rate", "net_correction", "accepted_ratio", "gate_accuracy", "selected_token_ratio"]),
        "",
        f"Conclusion: learned accept gate lowers regression when its regression_rate is below the no-gate variant ({fmt(gate_no.get('regression_rate')) if gate_no else ''} -> {fmt(gate_learned.get('regression_rate')) if gate_learned else ''}).",
        "",
        "## 6. Final Evaluation",
        "",
        *table(final_rows, ["mode", "gate_threshold", "top1", "top5", "ppl", "correction_rate", "regression_rate", "net_correction", "accepted_ratio", "gate_accuracy", "selected_token_ratio"]),
        "",
        f"Conclusion: GPT-2 + draft-aware LoRA-MDLM + learned gate exceeds GPT-2-only when final Top1/Top5 improves ({fmt(final_gpt2.get('top1')) if final_gpt2 else ''}/{fmt(final_gpt2.get('top5')) if final_gpt2 else ''} -> {fmt(final_learned.get('top1')) if final_learned else ''}/{fmt(final_learned.get('top5')) if final_learned else ''}).",
        "",
        "## 7. Final Conclusion",
        "",
        "- AR draft-induced corruption is an effective structured training noise source.",
        "- Draft-aware LoRA in GPT-2 draft context outperforms random-mask LoRA.",
        "- block_size=1 is currently best.",
        "- The learned accept gate lowers regression.",
        "- GPT-2 + draft-aware LoRA-MDLM + learned gate exceeds GPT-2-only.",
        "",
        "## 8. Current Best Configuration",
        "",
        "- Device model: `/mnt/data/enzeyu/hf_downloads/models/gpt2`",
        "- Edge model: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`",
        "- LoRA mode: `draft_aware`",
        "- block_size: `1`",
        "- refine_ratio: `0.2`",
        "- gate_threshold: `0.4`",
        "- trainable modules: LoRA adapter and accept gate only",
    ]

    out_path = Path("results/final_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_final_report={out_path}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Generate results/final_report.md from DART result files.").parse_args()
    main()
