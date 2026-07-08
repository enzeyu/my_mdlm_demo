"""Extended CoDraft-Diff evaluation, sweeps, analysis, and plotting."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F

from codraft_utils import (
    accept_gate,
    build_token_type_ids,
    cap_refine_mask_by_ratio,
    draft_aware_renoise,
    error_aware_score,
    estimate_draft_comm_mb,
    estimate_draft_risk,
    load_codraft_config,
    save_csv,
    save_json,
    select_top_ratio_by_score,
    teacher_forced_gpt2_draft_with_confidence,
    transition_counts,
    utility_accept_gate,
)
from data_real import build_dataloaders
from draft_utils import choose_device, load_gpt2, load_mdlm, pad_gpt2_logits, validate_model_surfaces
from eval_codraft_diff import load_eval_mdlm
from lora_utils import freeze_module
from metrics import format_table, now, sync_if_cuda


BASE_MODES = [
    "gpt2_only",
    "direct_draft_context",
    "random_refine_mask",
    "low_conf_refine",
    "draft_refine_adapter",
    "adapter_gate",
    "oracle_refine_mask",
]

RATIO_SWEEP_MODES = [
    "random_refine_mask",
    "low_conf_refine",
    "draft_refine_adapter",
    "adapter_gate",
    "oracle_refine_mask",
]

EXTENDED_MODES = [
    "error_aware_refine",
    "error_aware_adapter",
    "error_aware_adapter_gate",
    "utility_gate",
]

OUT_COLUMNS = [
    "mode",
    "seed",
    "requested_refine_ratio",
    "loss",
    "ppl",
    "final_acc",
    "top5",
    "correction_precision",
    "correction_recall",
    "wrong_edit_rate",
    "preserve",
    "refine_ratio",
    "accept_rate",
    "latency",
    "communication_MB",
    "wrong_to_correct",
    "correct_to_wrong",
    "wrong_to_wrong",
    "correct_to_correct",
    "edit_gain",
    "wrong_to_correct_ratio",
    "correct_to_wrong_ratio",
    "wrong_to_wrong_ratio",
    "correct_to_correct_ratio",
]


def _add_acc(acc: dict[str, Any], logits, labels, valid_mask, draft_ids, refined_ids, final_ids, refine_mask, accepted_mask, latency, comm_mb):
    safe_labels = labels.clamp(max=logits.size(-1) - 1)
    if bool(valid_mask.any()):
        selected_logits = logits[valid_mask]
        selected_labels = safe_labels[valid_mask]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
        top5 = selected_logits.topk(min(5, logits.size(-1)), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
        acc["loss_sum"] += float(loss.item())
        acc["top5_sum"] += int(top5.sum().item())
        acc["tokens"] += int(selected_labels.numel())
    counts = transition_counts(draft_ids, final_ids, labels, valid_mask)
    for key, value in counts.items():
        acc[key] += value
    changed = accepted_mask & final_ids.ne(draft_ids) & valid_mask
    good = changed & draft_ids.ne(labels) & final_ids.eq(labels)
    bad = changed & draft_ids.eq(labels) & final_ids.ne(labels)
    refined_correct = refined_ids.eq(labels) & refine_mask & valid_mask
    acc["changed"] += int(changed.sum().item())
    acc["good_changes"] += int(good.sum().item())
    acc["bad_changes"] += int(bad.sum().item())
    acc["draft_wrong"] += int((draft_ids.ne(labels) & valid_mask).sum().item())
    acc["draft_correct"] += int((draft_ids.eq(labels) & valid_mask).sum().item())
    acc["refined_correct"] += int(refined_correct.sum().item())
    acc["refine_tokens"] += int((refine_mask & valid_mask).sum().item())
    acc["accepted"] += int((accepted_mask & valid_mask).sum().item())
    acc["valid_tokens"] += int(valid_mask.sum().item())
    acc["latency"] += float(latency)
    acc["communication_MB"] += float(comm_mb)
    acc["batches"] += 1


def _finalize(mode: str, seed: int, ratio: float, acc: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(acc["tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    valid = max(int(acc["valid_tokens"]), 1)
    draft_wrong = max(int(acc["draft_wrong"]), 1)
    draft_correct = max(int(acc["draft_correct"]), 1)
    transition_total = max(
        int(acc["wrong_to_correct"] + acc["correct_to_wrong"] + acc["wrong_to_wrong"] + acc["correct_to_correct"]),
        1,
    )
    return {
        "mode": mode,
        "seed": seed,
        "requested_refine_ratio": ratio,
        "loss": loss,
        "ppl": float(math.exp(min(loss, 50.0))),
        "final_acc": acc["correct_to_correct"] / valid + acc["wrong_to_correct"] / valid,
        "top5": acc["top5_sum"] / tokens,
        "correction_precision": acc["good_changes"] / max(int(acc["changed"]), 1),
        "correction_recall": acc["good_changes"] / draft_wrong,
        "wrong_edit_rate": acc["bad_changes"] / draft_correct,
        "preserve": acc["correct_to_correct"] / draft_correct,
        "refine_ratio": acc["refine_tokens"] / valid,
        "accept_rate": acc["accepted"] / max(int(acc["refine_tokens"]), 1),
        "latency": acc["latency"] / max(int(acc["batches"]), 1),
        "communication_MB": acc["communication_MB"] / max(int(acc["batches"]), 1),
        "wrong_to_correct": int(acc["wrong_to_correct"]),
        "correct_to_wrong": int(acc["correct_to_wrong"]),
        "wrong_to_wrong": int(acc["wrong_to_wrong"]),
        "correct_to_correct": int(acc["correct_to_correct"]),
        "edit_gain": int(acc["wrong_to_correct"] - acc["correct_to_wrong"]),
        "wrong_to_correct_ratio": acc["wrong_to_correct"] / transition_total,
        "correct_to_wrong_ratio": acc["correct_to_wrong"] / transition_total,
        "wrong_to_wrong_ratio": acc["wrong_to_wrong"] / transition_total,
        "correct_to_correct_ratio": acc["correct_to_correct"] / transition_total,
    }


def _new_acc() -> defaultdict:
    return defaultdict(float)


class CoDraftExtendedEvaluator:
    def __init__(
        self,
        config_path: str,
        checkpoint: str,
        eval_steps: int | None = None,
        selector_profile: str = "confidence_entropy_margin",
        mdlm_disagreement_top_k: int = 5,
    ):
        self.config_path = config_path
        self.config = load_codraft_config(config_path)
        if eval_steps is not None:
            self.config["eval_steps"] = int(eval_steps)
        self.checkpoint = checkpoint
        self.selector_profile = selector_profile
        self.mdlm_disagreement_top_k = int(mdlm_disagreement_top_k)
        self.device = choose_device()
        _, self.val_loader, self.tokenizer, self.tokenizer_info = build_dataloaders(self.config)
        self.gpt2_model, _ = load_gpt2(self.config, self.tokenizer, self.device)
        freeze_module(self.gpt2_model)
        self.plain_model = load_eval_mdlm(self.config, self.tokenizer_info, self.device, None, adapter=False)
        validate_model_surfaces(self.plain_model, self.gpt2_model, self.tokenizer, self.tokenizer_info, self.device, int(self.config["max_length"]))
        self.adapter_model = load_eval_mdlm(self.config, self.tokenizer_info, self.device, checkpoint, adapter=True)
        validate_model_surfaces(self.adapter_model, self.gpt2_model, self.tokenizer, self.tokenizer_info, self.device, int(self.config["max_length"]))

    def _risk_batch(self, draft, clean, ratio: float):
        risk_scores, refine_mask, anchor_mask = estimate_draft_risk(
            draft["draft_ids"],
            draft["token_confidence"],
            draft["token_entropy"],
            draft["token_margin"],
            float(self.config.get("confidence_threshold", 0.5)),
            float(self.config.get("entropy_threshold", 3.0)),
            float(self.config.get("margin_threshold", 0.1)),
            expand_window=int(self.config.get("expand_window", 1)),
            min_span_len=int(self.config.get("min_span_len", 1)),
            pad_token_id=int(self.tokenizer_info.pad_token_id),
        )
        valid_mask = clean.ne(int(self.tokenizer_info.pad_token_id))
        refine_mask = cap_refine_mask_by_ratio(refine_mask, risk_scores, valid_mask, ratio)
        return risk_scores, refine_mask, anchor_mask, build_token_type_ids(refine_mask, anchor_mask)

    def _selector_score(self, draft, clean, valid_mask, ratio: float) -> torch.Tensor:
        profile = str(self.selector_profile)
        weights = {
            "confidence_only": (1.0, 0.0, 0.0, 0.0),
            "entropy_only": (0.0, 1.0, 0.0, 0.0),
            "margin_only": (0.0, 0.0, 1.0, 0.0),
            "confidence_entropy": (1.0, 1.0, 0.0, 0.0),
            "confidence_margin": (1.0, 0.0, 1.0, 0.0),
            "confidence_entropy_margin": (1.0, 1.0, 1.0, 0.0),
            "all_features": (1.0, 1.0, 1.0, 1.0),
            "mdlm_topk_disagreement_only": (0.0, 0.0, 0.0, 1.0),
            "gpt2_mdlm_disagreement_only": (0.0, 0.0, 0.0, 1.0),
        }.get(profile, (1.0, 1.0, 1.0, 0.0))
        disagreement = None
        if profile in {"all_features", "mdlm_topk_disagreement_only", "gpt2_mdlm_disagreement_only"}:
            timestep = torch.full((clean.size(0),), max(float(ratio), 1e-4), device=clean.device)
            logits = self.plain_model(draft["draft_ids"], timestep)["logits"].float()
            if profile == "mdlm_topk_disagreement_only":
                topk = logits.topk(min(self.mdlm_disagreement_top_k, logits.size(-1)), dim=-1).indices
                disagreement = ~topk.eq(draft["draft_ids"].unsqueeze(-1)).any(dim=-1)
            else:
                disagreement = logits.argmax(dim=-1).ne(draft["draft_ids"])
            disagreement = disagreement.float()
        return error_aware_score(
            draft["token_confidence"],
            draft["token_entropy"],
            draft["token_margin"],
            valid_mask,
            disagreement=disagreement,
            weights=weights,
        )

    def _select_mask(self, mode: str, draft, clean, valid_mask, low_conf_mask, risk_scores, ratio: float):
        if mode == "random_refine_mask":
            return select_random_ratio(valid_mask, ratio)
        if mode == "oracle_refine_mask":
            oracle_score = draft["draft_ids"].ne(clean).float()
            return select_top_ratio_by_score(oracle_score, valid_mask, ratio)
        if mode.startswith("error_aware"):
            score = self._selector_score(draft, clean, valid_mask, ratio)
            return select_top_ratio_by_score(score, valid_mask, ratio)
        if mode == "direct_draft_context":
            return valid_mask
        if mode == "gpt2_only":
            return torch.zeros_like(valid_mask)
        return low_conf_mask

    def _run_mode(self, mode, clean, valid_mask, draft, risk_scores, low_conf_mask, anchor_mask, token_type_ids, ratio: float):
        draft_ids = draft["draft_ids"]
        gpt2_logits = pad_gpt2_logits(draft["draft_logits"], int(self.tokenizer_info.vocab_size))
        if mode == "gpt2_only":
            empty = torch.zeros_like(valid_mask)
            return gpt2_logits, draft_ids, draft_ids, empty, empty, 0.0, empty

        refine_mask = self._select_mask(mode, draft, clean, valid_mask, low_conf_mask, risk_scores, ratio)
        if mode == "direct_draft_context":
            noisy_ids = draft_ids
            refine_mask = valid_mask
        else:
            noisy_ids, _, refine_mask = draft_aware_renoise(
                draft_ids,
                clean,
                refine_mask,
                anchor_mask,
                int(self.tokenizer_info.mask_token_id),
                pad_token_id=int(self.tokenizer_info.pad_token_id),
            )
        use_adapter = mode in {
            "draft_refine_adapter",
            "adapter_gate",
            "error_aware_adapter",
            "error_aware_adapter_gate",
            "error_aware_utility_gate",
            "utility_gate",
        }
        model = self.adapter_model if use_adapter else self.plain_model
        kwargs = {}
        if use_adapter:
            type_ids = build_token_type_ids(refine_mask, anchor_mask)
            kwargs = {
                "risk_scores": risk_scores,
                "token_confidence": draft["token_confidence"],
                "token_entropy": draft["token_entropy"],
                "token_margin": draft["token_margin"],
                "token_type_ids": type_ids,
            }
        timesteps = torch.full((clean.size(0),), max(float((refine_mask & valid_mask).float().mean().item()), 1e-4), device=clean.device)
        sync_if_cuda(clean.device)
        start = now()
        refined_logits = model(noisy_ids, timesteps, **kwargs)["logits"].float()
        sync_if_cuda(clean.device)
        latency = now() - start
        refined_ids = refined_logits.argmax(dim=-1)

        if mode in {"adapter_gate", "error_aware_adapter_gate"}:
            final_ids, accepted_mask, _, _ = accept_gate(
                draft_ids,
                refined_logits,
                refined_ids,
                refine_mask,
                draft["token_confidence"],
                risk_scores,
                accept_margin=float(self.config.get("accept_margin", 0.1)),
                accept_conf_threshold=float(self.config.get("accept_conf_threshold", 0.6)),
                risk_accept_threshold=float(self.config.get("risk_accept_threshold", 0.5)),
            )
        elif mode in {"utility_gate", "error_aware_utility_gate"}:
            final_ids, accepted_mask, _, _ = utility_accept_gate(
                draft_ids,
                refined_logits,
                refined_ids,
                refine_mask,
                draft["token_confidence"],
                utility_lambda=float(self.config.get("utility_lambda", 0.25)),
                utility_threshold=float(self.config.get("utility_threshold", 0.0)),
            )
        else:
            accepted_mask = refine_mask.clone()
            final_ids = draft_ids.clone()
            final_ids[accepted_mask] = refined_ids[accepted_mask]
        final_logits = gpt2_logits.clone()
        rows = accepted_mask.nonzero(as_tuple=False)
        if rows.numel() > 0:
            final_logits[rows[:, 0], rows[:, 1]] = refined_logits[rows[:, 0], rows[:, 1]]
        return final_logits, refined_ids, final_ids, accepted_mask & final_ids.ne(draft_ids), accepted_mask, latency, refine_mask

    @torch.no_grad()
    def evaluate(self, modes: list[str], ratio: float, seed: int = 7, collect_conf_bins: bool = False):
        torch.manual_seed(int(seed))
        accs = {mode: _new_acc() for mode in modes}
        bin_accs = {mode: [defaultdict(float) for _ in range(5)] for mode in modes} if collect_conf_bins else None
        for step, batch in enumerate(self.val_loader):
            if step >= int(self.config.get("eval_steps", 50)):
                break
            clean = batch.to(self.device, non_blocking=True)
            valid_mask = clean.ne(int(self.tokenizer_info.pad_token_id))
            sync_if_cuda(self.device)
            start = now()
            draft = teacher_forced_gpt2_draft_with_confidence(self.gpt2_model, clean, int(self.tokenizer_info.pad_token_id))
            draft["draft_ids"] = draft["draft_ids"].clamp(max=int(self.tokenizer_info.mask_token_id) - 1)
            sync_if_cuda(self.device)
            gpt2_latency = now() - start
            risk_scores, low_conf_mask, anchor_mask, token_type_ids = self._risk_batch(draft, clean, ratio)
            comm_mb = estimate_draft_comm_mb(clean.size(0), clean.size(1))
            for mode in modes:
                logits, refined_ids, final_ids, changed_mask, accepted_mask, latency, refine_mask = self._run_mode(
                    mode,
                    clean,
                    valid_mask,
                    draft,
                    risk_scores,
                    low_conf_mask,
                    anchor_mask,
                    token_type_ids,
                    ratio,
                )
                _add_acc(
                    accs[mode],
                    logits,
                    clean,
                    valid_mask,
                    draft["draft_ids"],
                    refined_ids,
                    final_ids,
                    refine_mask,
                    accepted_mask,
                    gpt2_latency + latency,
                    comm_mb,
                )
                if collect_conf_bins and bin_accs is not None:
                    add_confidence_bins(bin_accs[mode], draft, clean, valid_mask, final_ids, refine_mask, changed_mask)
            if step == 0 or (step + 1) % int(self.config.get("log_every", 100)) == 0:
                print(f"extended_eval step={step + 1}/{int(self.config.get('eval_steps', 50))} ratio={ratio} seed={seed}", flush=True)
        rows = [_finalize(mode, seed, ratio, accs[mode]) for mode in modes]
        bin_rows = finalize_confidence_bins(bin_accs) if bin_accs is not None else []
        return rows, bin_rows


def select_random_ratio(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    for row in range(valid_mask.size(0)):
        coords = torch.where(valid_mask[row])[0]
        if coords.numel() == 0:
            continue
        k = min(coords.numel(), max(1, int(math.ceil(coords.numel() * float(ratio)))))
        chosen = coords[torch.randperm(coords.numel(), device=coords.device)[:k]]
        selected[row, chosen] = True
    return selected & valid_mask


def add_confidence_bins(bin_accs, draft, labels, valid_mask, final_ids, refine_mask, changed_mask):
    conf = draft["token_confidence"]
    for idx in range(5):
        low = idx * 0.2
        high = 1.000001 if idx == 4 else (idx + 1) * 0.2
        mask = conf.ge(low) & conf.lt(high) & valid_mask
        if not bool(mask.any()):
            continue
        draft_correct = draft["draft_ids"].eq(labels) & mask
        draft_wrong = ~draft["draft_ids"].eq(labels) & mask
        final_correct = final_ids.eq(labels) & mask
        changed = changed_mask & mask
        good = changed & draft_wrong & final_correct
        bad = changed & draft_correct & ~final_correct
        acc = bin_accs[idx]
        acc["bin_low"] = low
        acc["bin_high"] = 1.0 if idx == 4 else high
        acc["tokens"] += int(mask.sum().item())
        acc["draft_errors"] += int(draft_wrong.sum().item())
        acc["refined"] += int((refine_mask & mask).sum().item())
        acc["changed"] += int(changed.sum().item())
        acc["good_changes"] += int(good.sum().item())
        acc["bad_changes"] += int(bad.sum().item())
        acc["draft_wrong"] += int(draft_wrong.sum().item())
        acc["draft_correct"] += int(draft_correct.sum().item())
        acc["preserved_correct"] += int((draft_correct & final_correct).sum().item())


def finalize_confidence_bins(all_bins) -> list[dict[str, Any]]:
    rows = []
    for mode, bins in all_bins.items():
        for idx, acc in enumerate(bins):
            tokens = max(int(acc["tokens"]), 1)
            draft_wrong = max(int(acc["draft_wrong"]), 1)
            draft_correct = max(int(acc["draft_correct"]), 1)
            rows.append(
                {
                    "mode": mode,
                    "bin": idx,
                    "bin_low": acc.get("bin_low", idx * 0.2),
                    "bin_high": acc.get("bin_high", (idx + 1) * 0.2),
                    "tokens": int(acc["tokens"]),
                    "draft_error_rate": acc["draft_errors"] / tokens,
                    "refined_token_ratio": acc["refined"] / tokens,
                    "correction_precision": acc["good_changes"] / max(int(acc["changed"]), 1),
                    "correction_recall": acc["good_changes"] / draft_wrong,
                    "wrong_edit_rate": acc["bad_changes"] / draft_correct,
                    "preserve": acc["preserved_correct"] / draft_correct,
                }
            )
    return rows


def plot_lines(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["mode"])].append(row)
    plt.figure(figsize=(7, 4.5))
    for mode, mode_rows in sorted(by_mode.items()):
        mode_rows = sorted(mode_rows, key=lambda item: float(item[x_key]))
        plt.plot([float(row[x_key]) for row in mode_rows], [float(row[y_key]) for row in mode_rows], marker="o", label=mode)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pr(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    for row in rows:
        plt.scatter(float(row["correction_recall"]), float(row["correction_precision"]), label=f"{row['mode']}@{row['requested_refine_ratio']}")
    plt.xlabel("correction_recall")
    plt.ylabel("correction_precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stacked_transitions(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["mode"]) for row in rows]
    keys = ["correct_to_correct_ratio", "correct_to_wrong_ratio", "wrong_to_correct_ratio", "wrong_to_wrong_ratio"]
    bottoms = [0.0] * len(rows)
    plt.figure(figsize=(9, 4.8))
    for key in keys:
        vals = [float(row[key]) for row in rows]
        plt.bar(labels, vals, bottom=bottoms, label=key)
        bottoms = [bottoms[idx] + vals[idx] for idx in range(len(vals))]
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("ratio")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confidence_bins(rows: list[dict[str, Any]], out_dir: Path) -> None:
    adapter_rows = [row for row in rows if row["mode"] in {"adapter_gate", "utility_gate"}]
    if not adapter_rows:
        adapter_rows = rows
    plot_lines(adapter_rows, "bin_low", "draft_error_rate", out_dir / "confidence_bin_error_rate.png", "Draft Error Rate by GPT-2 Confidence")
    plot_lines(adapter_rows, "bin_low", "correction_precision", out_dir / "confidence_bin_edit_quality.png", "Edit Quality by GPT-2 Confidence")


def sanity_check(rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        mode = row["mode"]
        if mode == "gpt2_only" and abs(float(row["refine_ratio"])) > 1e-9:
            warnings.append("gpt2_only refine_ratio is not zero")
        if mode not in {"gpt2_only", "direct_draft_context", "oracle_refine_mask"}:
            requested = float(row["requested_refine_ratio"])
            actual = float(row["refine_ratio"])
            if requested > 0 and abs(actual - requested) > max(0.025, requested * 0.2):
                warnings.append(f"{mode} requested ratio {requested} but got {actual}")
        if abs(float(row["preserve"]) - (1.0 - float(row["wrong_edit_rate"]))) > 1e-6:
            warnings.append(f"{mode} preserve != 1 - wrong_edit_rate")
    by_mode = {row["mode"]: row for row in rows}
    if "oracle_refine_mask" in by_mode and "random_refine_mask" in by_mode:
        if float(by_mode["oracle_refine_mask"]["final_acc"]) <= float(by_mode["random_refine_mask"]["final_acc"]):
            warnings.append("oracle_refine_mask did not beat random_refine_mask")
    return warnings


def run_ratio_sweep(args) -> list[dict[str, Any]]:
    evaluator = CoDraftExtendedEvaluator(args.config, args.checkpoint, args.eval_steps)
    out_dir = Path(args.out_dir) / "ratio_sweep"
    rows = []
    ratios = [float(value) for value in args.refine_ratios.split(",")]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for ratio in ratios:
        ratio_rows, _ = evaluator.evaluate(modes, ratio=ratio, seed=int(args.seed))
        rows.extend(ratio_rows)
        save_csv(out_dir / f"ratio_{ratio:.2f}.csv", ratio_rows, OUT_COLUMNS)
        save_json(out_dir / f"ratio_{ratio:.2f}.json", {"benchmark": ratio_rows, "warnings": sanity_check(ratio_rows)})
    save_csv(out_dir / "ratio_sweep_eval.csv", rows, OUT_COLUMNS)
    save_json(out_dir / "ratio_sweep_eval.json", {"benchmark": rows, "warnings": sanity_check(rows)})
    plot_lines(rows, "requested_refine_ratio", "final_acc", out_dir / "accuracy_vs_refine_ratio.png", "Accuracy vs Refinement Ratio")
    plot_lines(rows, "requested_refine_ratio", "wrong_edit_rate", out_dir / "wrong_edit_rate_vs_refine_ratio.png", "Wrong Edit Rate vs Refinement Ratio")
    plot_lines(rows, "requested_refine_ratio", "correction_recall", out_dir / "correction_recall_vs_refine_ratio.png", "Correction Recall vs Refinement Ratio")
    plot_pr(rows, out_dir / "precision_recall_tradeoff.png")
    return rows


def run_multiseed(args) -> list[dict[str, Any]]:
    evaluator = CoDraftExtendedEvaluator(args.config, args.checkpoint, args.eval_steps)
    out_dir = Path(args.out_dir) / "multiseed"
    seeds = [int(value) for value in args.seeds.split(",")]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    raw_rows = []
    for seed in seeds:
        rows, _ = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=seed)
        raw_rows.extend(rows)
        save_csv(out_dir / f"seed_{seed}.csv", rows, OUT_COLUMNS)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[row["mode"]].append(row)
    agg_rows = []
    metrics = ["final_acc", "top5", "correction_precision", "correction_recall", "wrong_edit_rate", "preserve", "refine_ratio"]
    for mode, mode_rows in grouped.items():
        out = {"mode": mode, "num_seeds": len(mode_rows), "requested_refine_ratio": float(args.refine_ratio)}
        for metric in metrics:
            vals = [float(row[metric]) for row in mode_rows]
            out[f"mean_{metric}"] = mean(vals)
            out[f"std_{metric}"] = pstdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(out)
    columns = ["mode", "num_seeds", "requested_refine_ratio"] + [f"{stat}_{metric}" for metric in metrics for stat in ("mean", "std")]
    save_csv(out_dir / "multiseed_eval.csv", agg_rows, columns)
    save_json(out_dir / "multiseed_eval.json", {"aggregate": agg_rows, "raw": raw_rows})
    return agg_rows


def run_analysis(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluator = CoDraftExtendedEvaluator(args.config, args.checkpoint, args.eval_steps)
    out_dir = Path(args.out_dir) / "analysis"
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows, bin_rows = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=int(args.seed), collect_conf_bins=True)
    save_csv(out_dir / "edit_transitions.csv", rows, OUT_COLUMNS)
    save_json(out_dir / "edit_transitions.json", {"benchmark": rows, "warnings": sanity_check(rows)})
    save_csv(out_dir / "confidence_bins.csv", bin_rows)
    save_json(out_dir / "confidence_bins.json", {"bins": bin_rows})
    plot_stacked_transitions(rows, out_dir / "edit_transition_stacked_bar.png")
    plot_confidence_bins(bin_rows, out_dir)
    return rows, bin_rows


def run_long_train_eval(args) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir) / "long_train"
    rows = []
    for step_value, checkpoint in parse_step_checkpoints(args.step_checkpoints).items():
        evaluator = CoDraftExtendedEvaluator(args.config, checkpoint, args.eval_steps)
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
        step_rows, _ = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=int(args.seed))
        for row in step_rows:
            row["training_steps"] = int(step_value)
        rows.extend(step_rows)
    columns = ["training_steps"] + OUT_COLUMNS
    save_csv(out_dir / "long_train_eval.csv", rows, columns)
    save_json(out_dir / "long_train_eval.json", {"benchmark": rows})
    plot_lines(rows, "training_steps", "final_acc", out_dir / "accuracy_vs_training_steps.png", "Accuracy vs Training Steps")
    plot_lines(rows, "training_steps", "wrong_edit_rate", out_dir / "wrong_edit_rate_vs_training_steps.png", "Wrong Edit Rate vs Training Steps")
    plot_lines(rows, "training_steps", "correction_precision", out_dir / "correction_precision_recall_vs_training_steps.png", "Precision vs Training Steps")
    return rows


def run_best_10000(args) -> list[dict[str, Any]]:
    evaluator = CoDraftExtendedEvaluator(
        args.config,
        args.checkpoint,
        args.eval_steps,
        selector_profile=args.selector_profile,
        mdlm_disagreement_top_k=args.mdlm_disagreement_top_k,
    )
    out_dir = Path(args.out_dir)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows, _ = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=int(args.seed))
    save_csv(out_dir / "best_10000_eval.csv", rows, OUT_COLUMNS)
    save_json(out_dir / "best_10000_eval.json", {"benchmark": rows, "warnings": sanity_check(rows), "checkpoint": args.checkpoint})
    return rows


def run_ratio_sweep_10000(args) -> list[dict[str, Any]]:
    evaluator = CoDraftExtendedEvaluator(
        args.config,
        args.checkpoint,
        args.eval_steps,
        selector_profile=args.selector_profile,
        mdlm_disagreement_top_k=args.mdlm_disagreement_top_k,
    )
    out_dir = Path(args.out_dir)
    rows = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for ratio in [float(value) for value in args.refine_ratios.split(",")]:
        ratio_rows, _ = evaluator.evaluate(modes, ratio=ratio, seed=int(args.seed))
        rows.extend(ratio_rows)
    save_csv(out_dir / "ratio_sweep_10000.csv", rows, OUT_COLUMNS)
    save_json(out_dir / "ratio_sweep_10000.json", {"benchmark": rows, "warnings": sanity_check(rows), "checkpoint": args.checkpoint})
    plot_lines(rows, "requested_refine_ratio", "final_acc", out_dir / "accuracy_vs_ratio_10000.png", "Accuracy vs Ratio, 10000-step")
    plot_lines(rows, "requested_refine_ratio", "wrong_edit_rate", out_dir / "wrong_edit_vs_ratio_10000.png", "Wrong Edit vs Ratio, 10000-step")
    plot_lines(rows, "requested_refine_ratio", "edit_gain", out_dir / "edit_gain_vs_ratio_10000.png", "Edit Gain vs Ratio, 10000-step")
    plot_pr(rows, out_dir / "precision_recall_tradeoff_10000.png")
    return rows


def run_gate_threshold_sweep(args) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    rows = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for threshold in [float(value) for value in args.gate_thresholds.split(",")]:
        evaluator = CoDraftExtendedEvaluator(
            args.config,
            args.checkpoint,
            args.eval_steps,
            selector_profile=args.selector_profile,
            mdlm_disagreement_top_k=args.mdlm_disagreement_top_k,
        )
        evaluator.config["accept_margin"] = threshold
        evaluator.config["utility_threshold"] = threshold
        threshold_rows, _ = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=int(args.seed))
        for row in threshold_rows:
            row["gate_threshold"] = threshold
        rows.extend(threshold_rows)
    columns = ["gate_threshold", *OUT_COLUMNS]
    save_csv(out_dir / "gate_threshold_sweep.csv", rows, columns)
    save_json(out_dir / "gate_threshold_sweep.json", {"benchmark": rows, "warnings": sanity_check(rows), "checkpoint": args.checkpoint})
    plot_lines(rows, "gate_threshold", "final_acc", out_dir / "final_acc_vs_gate_threshold.png", "Final Accuracy vs Gate Threshold")
    plot_lines(rows, "gate_threshold", "wrong_edit_rate", out_dir / "wrong_edit_vs_gate_threshold.png", "Wrong Edit vs Gate Threshold")
    plot_lines(rows, "gate_threshold", "edit_gain", out_dir / "edit_gain_vs_gate_threshold.png", "Edit Gain vs Gate Threshold")
    plot_transition_lines(rows, out_dir / "wtc_ctw_vs_gate_threshold.png")
    return rows


def run_selector_ablation(args) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    rows = []
    profiles = [profile.strip() for profile in args.selector_profiles.split(",") if profile.strip()]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for profile in profiles:
        evaluator = CoDraftExtendedEvaluator(
            args.config,
            args.checkpoint,
            args.eval_steps,
            selector_profile=profile,
            mdlm_disagreement_top_k=args.mdlm_disagreement_top_k,
        )
        profile_rows, _ = evaluator.evaluate(modes, ratio=float(args.refine_ratio), seed=int(args.seed))
        for row in profile_rows:
            row["selector_profile"] = profile
        rows.extend(profile_rows)
    columns = ["selector_profile", *OUT_COLUMNS]
    save_csv(out_dir / "selector_ablation.csv", rows, columns)
    save_json(out_dir / "selector_ablation.json", {"benchmark": rows, "checkpoint": args.checkpoint})
    plot_bar(rows, "selector_profile", "final_acc", out_dir / "selector_ablation_acc.png", "Selector Ablation Accuracy")
    plot_bar(rows, "selector_profile", "edit_gain", out_dir / "selector_ablation_edit_gain.png", "Selector Ablation Edit Gain")
    plot_bar(rows, "selector_profile", "wrong_edit_rate", out_dir / "selector_ablation_wrong_edit.png", "Selector Ablation Wrong Edit")
    return rows


def plot_transition_lines(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    for mode in sorted({str(row["mode"]) for row in rows}):
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda row: float(row["gate_threshold"]))
        plt.plot([float(row["gate_threshold"]) for row in mode_rows], [float(row["wrong_to_correct"]) for row in mode_rows], marker="o", label=f"{mode} W->C")
        plt.plot([float(row["gate_threshold"]) for row in mode_rows], [float(row["correct_to_wrong"]) for row in mode_rows], marker="x", linestyle="--", label=f"{mode} C->W")
    plt.xlabel("gate_threshold")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if str(row["mode"]).endswith("gate") or str(row["mode"]).endswith("utility_gate")]
    if not plot_rows:
        plot_rows = rows
    labels = [f"{row[x_key]}\n{row['mode']}" for row in plot_rows]
    values = [float(row[y_key]) for row in plot_rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 0.7), 4.5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_best_10000_readme(out_dir: str, decision: dict[str, Any]) -> None:
    path = Path(out_dir) / "README.md"
    lines = [
        "# Best 10000-step CoDraft-Diff Evaluation",
        "",
        f"Checkpoint: `{decision.get('checkpoint', '')}`",
        "",
        "## Experiments",
        "",
        "- `best_10000_eval.csv/json`: ratio=0.20 method comparison.",
        "- `ratio_sweep_10000.csv/json`: refinement ratio sweep.",
        "- `gate_threshold_sweep.csv/json`: adapter/utility gate threshold sweep.",
        "- `selector_ablation.csv/json`: selector feature ablation.",
        "- `next_step_decision.md`: recommendation for whether to run 30000-step training.",
        "",
        "## Plots",
        "",
        "- `accuracy_vs_ratio_10000.png`: accuracy across refinement ratios.",
        "- `wrong_edit_vs_ratio_10000.png`: harmful edit rate across ratios.",
        "- `edit_gain_vs_ratio_10000.png`: W->C minus C->W across ratios.",
        "- `precision_recall_tradeoff_10000.png`: correction precision/recall tradeoff.",
        "- `final_acc_vs_gate_threshold.png`, `wrong_edit_vs_gate_threshold.png`, `edit_gain_vs_gate_threshold.png`, `wtc_ctw_vs_gate_threshold.png`: gate sweep plots.",
        "- `selector_ablation_acc.png`, `selector_ablation_edit_gain.png`, `selector_ablation_wrong_edit.png`: selector feature ablation plots.",
        "",
        f"Current best non-oracle: `{decision.get('best_method', '')}` final_acc={decision.get('best_final_acc', 0):.4f}.",
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_step_decision(args) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    best_rows = load_rows(out_dir / "best_10000_eval.csv")
    ratio_rows = load_rows(out_dir / "ratio_sweep_10000.csv")
    gate_rows = load_rows(out_dir / "gate_threshold_sweep.csv")
    selector_rows = load_rows(out_dir / "selector_ablation.csv")
    all_rows = best_rows + ratio_rows + gate_rows + selector_rows
    non_oracle = [row for row in all_rows if row["mode"] != "oracle_refine_mask"]
    best = max(non_oracle, key=lambda row: float(row["final_acc"]))
    oracle_candidates = [
        row for row in all_rows
        if row["mode"] == "oracle_refine_mask"
        and abs(float(row["requested_refine_ratio"]) - float(best["requested_refine_ratio"])) < 1e-9
    ]
    oracle = max(oracle_candidates or [row for row in all_rows if row["mode"] == "oracle_refine_mask"], key=lambda row: float(row["final_acc"]))
    previous_best = 0.3243
    improvement = float(best["final_acc"]) - previous_best
    oracle_gap = float(oracle["final_acc"]) - float(best["final_acc"])
    recommend = float(best["final_acc"]) >= 0.327 or improvement >= 0.005
    priorities = []
    if float(best["wrong_edit_rate"]) > 0.035:
        priorities.append("gate")
    if oracle_gap > 0.01:
        priorities.append("selector")
    if float(best["correction_precision"]) < 0.20:
        priorities.append("adapter loss")
    if not priorities:
        priorities.append("joint selector/gate tuning")
    recommendation = (
        "Start 30000-step training."
        if recommend
        else "Do not start 30000-step yet; prioritize " + ", ".join(priorities) + "."
    )
    decision = {
        "checkpoint": args.checkpoint,
        "best_method": best["mode"],
        "best_ratio": float(best["requested_refine_ratio"]),
        "best_selector_profile": best.get("selector_profile", ""),
        "best_gate_threshold": best.get("gate_threshold", ""),
        "best_final_acc": float(best["final_acc"]),
        "best_wrong_edit_rate": float(best["wrong_edit_rate"]),
        "best_correction_precision": float(best["correction_precision"]),
        "previous_3000_best_final_acc": previous_best,
        "improvement_over_3000_best": improvement,
        "oracle_final_acc": float(oracle["final_acc"]),
        "oracle_gap": oracle_gap,
        "recommend_30000": recommend,
        "recommendation": recommendation,
        "priorities": priorities,
    }
    lines = [
        "# Next Step Decision",
        "",
        f"1. Current 10000-step best non-oracle: `{best['mode']}` at ratio `{float(best['requested_refine_ratio']):.2f}`.",
        f"   selector_profile: `{best.get('selector_profile', '')}`; gate_threshold: `{best.get('gate_threshold', '')}`.",
        f"2. Best non-oracle final_acc >= 0.326? {'yes' if float(best['final_acc']) >= 0.326 else 'no'} ({float(best['final_acc']):.4f}).",
        f"3. Improvement over 3000-step best >= 0.003? {'yes' if improvement >= 0.003 else 'no'} ({improvement:.4f}).",
        f"4. Oracle gap: `{oracle_gap:.4f}`.",
        f"5. wrong_edit_rate < 0.03? {'yes' if float(best['wrong_edit_rate']) < 0.03 else 'no'} ({float(best['wrong_edit_rate']):.4f}).",
        f"6. Recommend 30000-step training? {'yes' if recommend else 'no'}.",
        "",
        f"Recommendation: {recommendation}",
        "",
        f"Best correction_precision: `{float(best['correction_precision']):.4f}`.",
        f"Priority: `{', '.join(priorities)}`.",
    ]
    (out_dir / "next_step_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(out_dir / "next_step_decision.json", decision)
    write_best_10000_readme(str(out_dir), decision)
    return decision


def load_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_step_checkpoints(text: str) -> dict[int, str]:
    mapping = {}
    for item in text.split(","):
        if not item.strip():
            continue
        step, path = item.split(":", 1)
        mapping[int(step)] = path
    return mapping


def write_readme(out_dir: str) -> None:
    path = Path(out_dir) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# CoDraft-Diff GPU1 Extended Experiments",
                "",
                "This directory contains extended evaluation outputs for AR-Diffusion edge collaboration.",
                "",
                "## Scripts",
                "",
                "- `scripts/run_codraft_ratio_sweep.sh`: runs refinement-ratio sweep and writes `ratio_sweep/`.",
                "- `scripts/run_codraft_multiseed.sh`: runs seed aggregation and writes `multiseed/`.",
                "- `scripts/run_codraft_long_train.sh`: trains/evaluates longer checkpoints when requested.",
                "- `scripts/analyze_codraft_results.py`: runs confidence-bin and edit-transition analysis.",
                "- `scripts/plot_codraft_results.py`: regenerates plots from saved CSV files.",
                "",
                "## Inputs",
                "",
                "- Config: `configs/codraft_diff_wikitext2_gpu1.yaml`",
                "- Adapter checkpoint: `results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt`",
                "",
                "## Outputs",
                "",
                "- `ratio_sweep/ratio_sweep_eval.csv` and `.json`",
                "- `multiseed/multiseed_eval.csv` and `.json`",
                "- `analysis/confidence_bins.csv` and `.json`",
                "- `analysis/edit_transitions.csv` and `.json`",
                "- plot PNG files in corresponding subdirectories",
                "",
                "All outputs are written under `results/codraft_diff_gpu1_extended/`; the original `results/codraft_diff_gpu1/` directory is not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/codraft_diff_wikitext2_gpu1.yaml")
    common.add_argument("--checkpoint", default="results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt")
    common.add_argument("--out_dir", default="results/codraft_diff_gpu1_extended")
    common.add_argument("--eval_steps", type=int, default=200)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--selector_profile", default="confidence_entropy_margin")
    common.add_argument("--mdlm_disagreement_top_k", type=int, default=5)

    ratio = sub.add_parser("ratio_sweep", parents=[common])
    ratio.add_argument("--refine_ratios", default="0.05,0.10,0.20,0.30,0.40,0.60")
    ratio.add_argument("--modes", default="random_refine_mask,low_conf_refine,draft_refine_adapter,adapter_gate,oracle_refine_mask,error_aware_refine,error_aware_adapter,error_aware_adapter_gate,utility_gate")

    multi = sub.add_parser("multiseed", parents=[common])
    multi.add_argument("--seeds", default="0,1,2")
    multi.add_argument("--refine_ratio", type=float, default=0.20)
    multi.add_argument("--modes", default="gpt2_only,direct_draft_context,random_refine_mask,low_conf_refine,draft_refine_adapter,adapter_gate,oracle_refine_mask")

    analysis = sub.add_parser("analysis", parents=[common])
    analysis.add_argument("--refine_ratio", type=float, default=0.20)
    analysis.add_argument("--modes", default="gpt2_only,direct_draft_context,random_refine_mask,low_conf_refine,draft_refine_adapter,adapter_gate,oracle_refine_mask,error_aware_refine,error_aware_adapter,error_aware_adapter_gate,utility_gate")

    long_train = sub.add_parser("long_train_eval", parents=[common])
    long_train.add_argument("--refine_ratio", type=float, default=0.20)
    long_train.add_argument("--modes", default="low_conf_refine,draft_refine_adapter,adapter_gate,utility_gate")
    long_train.add_argument("--step_checkpoints", default="3000:results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt")

    best = sub.add_parser("best_10000", parents=[common])
    best.set_defaults(
        checkpoint="results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt",
        out_dir="results/codraft_diff_gpu1_extended/best_10000_eval",
    )
    best.add_argument("--refine_ratio", type=float, default=0.20)
    best.add_argument("--modes", default="draft_refine_adapter,adapter_gate,error_aware_refine,error_aware_adapter,error_aware_adapter_gate,utility_gate,error_aware_utility_gate,random_refine_mask,oracle_refine_mask")

    ratio10000 = sub.add_parser("ratio_sweep_10000", parents=[common])
    ratio10000.set_defaults(
        checkpoint="results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt",
        out_dir="results/codraft_diff_gpu1_extended/best_10000_eval",
    )
    ratio10000.add_argument("--refine_ratios", default="0.05,0.10,0.20,0.30,0.40,0.60")
    ratio10000.add_argument("--modes", default="random_refine_mask,low_conf_refine,adapter_gate,error_aware_adapter_gate,utility_gate,error_aware_utility_gate,oracle_refine_mask")

    gate_sweep = sub.add_parser("gate_threshold_sweep", parents=[common])
    gate_sweep.set_defaults(
        checkpoint="results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt",
        out_dir="results/codraft_diff_gpu1_extended/best_10000_eval",
    )
    gate_sweep.add_argument("--refine_ratio", type=float, default=0.20)
    gate_sweep.add_argument("--gate_thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    gate_sweep.add_argument("--modes", default="adapter_gate,error_aware_adapter_gate,utility_gate,error_aware_utility_gate")

    selector_ablation = sub.add_parser("selector_ablation", parents=[common])
    selector_ablation.set_defaults(
        checkpoint="results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt",
        out_dir="results/codraft_diff_gpu1_extended/best_10000_eval",
    )
    selector_ablation.add_argument("--refine_ratio", type=float, default=0.20)
    selector_ablation.add_argument("--selector_profiles", default="confidence_only,entropy_only,margin_only,mdlm_topk_disagreement_only,gpt2_mdlm_disagreement_only,confidence_entropy,confidence_margin,confidence_entropy_margin,all_features")
    selector_ablation.add_argument("--modes", default="error_aware_adapter,error_aware_adapter_gate,error_aware_utility_gate")

    decision = sub.add_parser("decision", parents=[common])
    decision.set_defaults(
        checkpoint="results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt",
        out_dir="results/codraft_diff_gpu1_extended/best_10000_eval",
    )

    readme = sub.add_parser("write_readme")
    readme.add_argument("--out_dir", default="results/codraft_diff_gpu1_extended")

    args = parser.parse_args()
    if args.command == "ratio_sweep":
        rows = run_ratio_sweep(args)
        print(format_table(rows, OUT_COLUMNS[:15]))
    elif args.command == "multiseed":
        rows = run_multiseed(args)
        print(json.dumps(rows, indent=2))
    elif args.command == "analysis":
        rows, _ = run_analysis(args)
        print(format_table(rows, OUT_COLUMNS[:15]))
    elif args.command == "long_train_eval":
        rows = run_long_train_eval(args)
        print(format_table(rows, ["training_steps", *OUT_COLUMNS[:15]]))
    elif args.command == "best_10000":
        rows = run_best_10000(args)
        print(format_table(rows, OUT_COLUMNS[:20]))
    elif args.command == "ratio_sweep_10000":
        rows = run_ratio_sweep_10000(args)
        print(format_table(rows, OUT_COLUMNS[:20]))
    elif args.command == "gate_threshold_sweep":
        rows = run_gate_threshold_sweep(args)
        print(format_table(rows, ["gate_threshold", *OUT_COLUMNS[:20]]))
    elif args.command == "selector_ablation":
        rows = run_selector_ablation(args)
        print(format_table(rows, ["selector_profile", *OUT_COLUMNS[:20]]))
    elif args.command == "decision":
        decision_payload = write_next_step_decision(args)
        print(json.dumps(decision_payload, indent=2))
    elif args.command == "write_readme":
        write_readme(args.out_dir)


if __name__ == "__main__":
    main()
