"""Generate the final Draft-aware LoRA analysis report from saved result files."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from(payload) -> list[dict]:
    if not payload:
        return []
    return payload.get("benchmark", payload if isinstance(payload, list) else [])


def fmt(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
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
    training_mode = rows_from(load_json(Path("results/lora_training_mode_comparison/training_mode_comparison.json")))
    block_size = rows_from(load_json(Path("results/block_size_ablation/block_size_ablation.json")))
    accept_gate = rows_from(load_json(Path("results/accept_gate/accept_gate_eval.json")))

    random_row = next((row for row in training_mode if row.get("Train Noise") == "random_mask"), None)
    draft_row = next((row for row in training_mode if row.get("Train Noise") == "draft_aware"), None)
    mode_support = bool(random_row and draft_row and draft_row.get("Draft Top1", 0) > random_row.get("Draft Top1", 0))

    token_row = next((row for row in block_size if int(row.get("Block Size", -1)) == 1), None)
    block4_row = next((row for row in block_size if int(row.get("Block Size", -1)) == 4), None)
    best_block = best(block_size, "Draft Top1")

    gpt2 = next((row for row in accept_gate if row.get("mode") == "gpt2_only"), None)
    learned_rows = [row for row in accept_gate if row.get("mode") == "draft_aware_lora_refine_with_learned_gate"]
    learned = max(learned_rows, key=lambda row: (row.get("top1", 0), row.get("net_correction", -99))) if learned_rows else None
    no_gate = next((row for row in accept_gate if row.get("mode") == "draft_aware_lora_refine_no_gate"), None)

    lines = [
        "# Draft-aware LoRA Fine-tuning for Edge MDLM",
        "",
        "## 1. Problem",
        "",
        "Pretrained MDLM is strong under clean standard random-mask denoising, but the existing experiments show a clear drop when the visible context is a GPT-2 draft rather than clean tokens. This mismatch matters for edge-device collaboration because the edge MDLM must refine structured AR draft errors, not independent random masks.",
        "",
        "## 2. Idea",
        "",
        "AR draft-induced corruption is treated as a structured training noise source: GPT-2 produces a draft context, low-confidence tokens or blocks are masked, and LoRA trains the frozen MDLM to recover the clean tokens from that draft context. GPT-2 is not trained and the full MDLM checkpoint is not saved.",
        "",
        "## 3. Random-mask LoRA vs Draft-aware LoRA",
        "",
        *markdown_table(
            training_mode,
            ["Model", "Train Noise", "Standard Top1", "Standard Top5", "Draft Top1", "Draft Top5", "Draft PPL", "Correction", "Regression", "Net"],
        ),
        "",
        f"Draft-aware LoRA beats random-mask LoRA on GPT-2 draft-context Top1: {'yes' if mode_support else 'no or not enough evidence'}.",
        "",
        "## 4. Block-size Ablation",
        "",
        *markdown_table(
            block_size,
            ["Block Size", "Draft Top1", "Draft Top5", "PPL", "Block EM", "Correction", "Regression", "Net", "Selected Ratio"],
        ),
        "",
        f"block_size=4 vs token-level block_size=1: {'better' if token_row and block4_row and block4_row.get('Draft Top1', 0) > token_row.get('Draft Top1', 0) else 'not better or not enough evidence'}.",
        f"Current best block_size by Draft Top1: `{best_block.get('Block Size') if best_block else ''}`.",
        "",
        "## 5. Accept Gate",
        "",
        *markdown_table(
            accept_gate,
            ["mode", "gate_threshold", "top1", "top5", "ppl", "correction_rate", "regression_rate", "net_correction", "accepted_ratio", "gate_accuracy", "selected_token_ratio"],
        ),
        "",
        f"Learned gate lowers regression vs no gate: {'yes' if learned and no_gate and learned.get('regression_rate', 1) < no_gate.get('regression_rate', 0) else 'no or not enough evidence'}.",
        f"Learned gate exceeds GPT-2-only Top1/Top5: {'yes' if learned and gpt2 and (learned.get('top1', 0), learned.get('top5', 0)) > (gpt2.get('top1', 0), gpt2.get('top5', 0)) else 'no or not enough evidence'}.",
        f"Best learned-gate threshold: `{learned.get('gate_threshold') if learned else ''}`.",
        "",
        "## 6. Final Conclusion",
        "",
        "The conclusion should be read from the three result groups above. The hypothesis is supported when draft-aware LoRA outperforms random-mask LoRA under GPT-2 draft context, block-level training improves over token-level or avoids degradation, and the accept gate reduces harmful regressions enough for GPT-2 + LoRA-MDLM refinement to match or exceed GPT-2-only.",
        "",
        f"Current aggregate support: {'supported' if mode_support and learned and gpt2 and (learned.get('top1', 0), learned.get('top5', 0)) > (gpt2.get('top1', 0), gpt2.get('top5', 0)) else 'partial or not yet sufficient'}.",
    ]
    out_path = Path("results/draft_aware_lora_final_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_final_report={out_path}")


if __name__ == "__main__":
    main()
