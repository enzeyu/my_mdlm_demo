"""Evaluate CoDraft-Diff collaboration modes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from codraft_utils import (
    accept_gate,
    build_token_type_ids,
    cap_refine_mask_by_ratio,
    correction_metrics,
    draft_aware_renoise,
    estimate_draft_comm_mb,
    estimate_draft_risk,
    load_codraft_config,
    save_csv,
    save_json,
    teacher_forced_gpt2_draft_with_confidence,
)
from data_real import build_dataloaders, mask_tokens
from draft_utils import choose_device, load_gpt2, load_mdlm, pad_gpt2_logits, validate_model_surfaces
from lora_utils import freeze_module
from metrics import format_table, now, sync_if_cuda


COLUMNS = [
    "mode",
    "loss",
    "ppl",
    "draft_token_acc",
    "refined_token_acc",
    "final_token_acc",
    "top5_acc",
    "correction_precision",
    "correction_recall",
    "wrong_edit_rate",
    "correct_token_preservation",
    "refine_ratio",
    "accept_rate",
    "latency",
    "communication_MB",
]


def new_acc() -> dict:
    return {
        "loss_sum": 0.0,
        "top5_sum": 0,
        "tokens": 0,
        "draft_correct": 0,
        "refined_correct": 0,
        "refined_tokens": 0,
        "final_correct": 0,
        "good_changes": 0,
        "changed": 0,
        "draft_wrong": 0,
        "bad_changes": 0,
        "draft_correct_total": 0,
        "preserved_correct": 0,
        "refine_tokens": 0,
        "valid_tokens": 0,
        "accepted": 0,
        "latency": 0.0,
        "communication_MB": 0.0,
        "batches": 0,
    }


def add_row_metrics(
    acc: dict,
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    draft_ids: torch.Tensor,
    refined_ids: torch.Tensor,
    final_ids: torch.Tensor,
    refine_mask: torch.Tensor,
    changed_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    latency: float,
    communication_mb: float,
) -> None:
    safe_labels = labels.clamp(max=logits.size(-1) - 1)
    selected_logits = logits[valid_mask]
    selected_labels = safe_labels[valid_mask]
    if selected_labels.numel() > 0:
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
        top5 = selected_logits.topk(min(5, logits.size(-1)), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
        acc["loss_sum"] += float(loss.item())
        acc["top5_sum"] += int(top5.sum().item())
        acc["tokens"] += int(selected_labels.numel())

    metrics = correction_metrics(draft_ids, refined_ids, final_ids, labels, valid_mask, changed_mask, refine_mask)
    valid = int(valid_mask.sum().item())
    refined_count = int((refine_mask & valid_mask).sum().item())
    draft_correct = draft_ids.eq(labels) & valid_mask
    draft_wrong = ~draft_ids.eq(labels) & valid_mask
    final_correct = final_ids.eq(labels) & valid_mask
    changed = changed_mask & valid_mask
    good_changes = changed & draft_wrong & final_correct
    bad_changes = changed & draft_correct & ~final_correct

    acc["draft_correct"] += int((draft_ids.eq(labels) & valid_mask).sum().item())
    acc["refined_correct"] += int((refined_ids.eq(labels) & refine_mask & valid_mask).sum().item())
    acc["refined_tokens"] += refined_count
    acc["final_correct"] += int(final_correct.sum().item())
    acc["good_changes"] += int(good_changes.sum().item())
    acc["changed"] += int(changed.sum().item())
    acc["draft_wrong"] += int(draft_wrong.sum().item())
    acc["bad_changes"] += int(bad_changes.sum().item())
    acc["draft_correct_total"] += int(draft_correct.sum().item())
    acc["preserved_correct"] += int((draft_correct & final_correct).sum().item())
    acc["refine_tokens"] += refined_count
    acc["valid_tokens"] += valid
    acc["accepted"] += int((accepted_mask & valid_mask).sum().item())
    acc["latency"] += float(latency)
    acc["communication_MB"] += float(communication_mb)
    acc["batches"] += 1
    del metrics


def finalize(mode: str, acc: dict) -> dict:
    tokens = max(int(acc["tokens"]), 1)
    loss = acc["loss_sum"] / tokens
    return {
        "mode": mode,
        "loss": loss,
        "ppl": float(math.exp(min(loss, 50.0))),
        "draft_token_acc": acc["draft_correct"] / max(acc["valid_tokens"], 1),
        "refined_token_acc": acc["refined_correct"] / max(acc["refined_tokens"], 1),
        "final_token_acc": acc["final_correct"] / max(acc["valid_tokens"], 1),
        "top5_acc": acc["top5_sum"] / tokens,
        "correction_precision": acc["good_changes"] / max(acc["changed"], 1),
        "correction_recall": acc["good_changes"] / max(acc["draft_wrong"], 1),
        "wrong_edit_rate": acc["bad_changes"] / max(acc["draft_correct_total"], 1),
        "correct_token_preservation": acc["preserved_correct"] / max(acc["draft_correct_total"], 1),
        "refine_ratio": acc["refine_tokens"] / max(acc["valid_tokens"], 1),
        "accept_rate": acc["accepted"] / max(acc["refine_tokens"], 1),
        "latency": acc["latency"] / max(acc["batches"], 1),
        "communication_MB": acc["communication_MB"] / max(acc["batches"], 1),
    }


def load_eval_mdlm(config: dict, tokenizer_info, device: torch.device, checkpoint: str | None, adapter: bool):
    cfg = dict(config)
    if adapter:
        cfg["use_draft_adapter"] = True
        cfg["use_draft_conditioning"] = True
    else:
        cfg["use_draft_adapter"] = False
        cfg["use_draft_conditioning"] = False
    model = load_mdlm(cfg, tokenizer_info, device, None).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"checkpoint_load_warning adapter={adapter} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    freeze_module(model)
    return model


def select_random_like(refine_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(refine_mask)
    for row in range(refine_mask.size(0)):
        k = int((refine_mask[row] & valid_mask[row]).sum().item())
        coords = torch.where(valid_mask[row])[0]
        if k > 0 and coords.numel() > 0:
            chosen = coords[torch.randperm(coords.numel(), device=coords.device)[: min(k, coords.numel())]]
            selected[row, chosen] = True
    return selected


def build_risk_batch(config, draft, clean, tokenizer_info):
    risk_scores, refine_mask, anchor_mask = estimate_draft_risk(
        draft["draft_ids"],
        draft["token_confidence"],
        draft["token_entropy"],
        draft["token_margin"],
        float(config.get("confidence_threshold", 0.5)),
        float(config.get("entropy_threshold", 3.0)),
        float(config.get("margin_threshold", 0.1)),
        expand_window=int(config.get("expand_window", 1)),
        min_span_len=int(config.get("min_span_len", 1)),
        pad_token_id=int(tokenizer_info.pad_token_id),
    )
    valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
    refine_mask = cap_refine_mask_by_ratio(
        refine_mask,
        risk_scores,
        valid_mask,
        float(config.get("refine_ratio", 0.2)),
    )
    token_type_ids = build_token_type_ids(refine_mask, anchor_mask)
    return risk_scores, refine_mask, anchor_mask, token_type_ids


@torch.no_grad()
def run_mode_batch(
    mode: str,
    model,
    clean: torch.Tensor,
    valid_mask: torch.Tensor,
    draft: dict,
    risk_scores: torch.Tensor,
    low_conf_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    tokenizer_info,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    draft_ids = draft["draft_ids"]
    refined_ids = draft_ids.clone()
    final_ids = draft_ids.clone()
    changed_mask = torch.zeros_like(valid_mask)
    accepted_mask = torch.zeros_like(valid_mask)
    gpt2_logits = pad_gpt2_logits(draft["draft_logits"], int(tokenizer_info.vocab_size))

    if mode == "gpt2_only":
        return gpt2_logits, refined_ids, final_ids, changed_mask, accepted_mask, 0.0

    if mode == "direct_draft_context":
        noisy_ids = draft_ids
        refine_mask = valid_mask
        model_risk = model_draft = model_types = None
    elif mode in {"low_conf_refine", "draft_refine_adapter", "draft_refine_adapter_gate"}:
        refine_mask = low_conf_mask
        noisy_ids, _, refine_mask = draft_aware_renoise(
            draft_ids,
            clean,
            refine_mask,
            anchor_mask,
            int(tokenizer_info.mask_token_id),
            pad_token_id=int(tokenizer_info.pad_token_id),
        )
        model_risk = risk_scores if mode.startswith("draft_refine_adapter") else None
        model_draft = draft if mode.startswith("draft_refine_adapter") else None
        model_types = token_type_ids if mode.startswith("draft_refine_adapter") else None
    elif mode == "random_refine_mask":
        refine_mask = select_random_like(low_conf_mask, valid_mask)
        noisy_ids, _, refine_mask = draft_aware_renoise(draft_ids, clean, refine_mask, anchor_mask, int(tokenizer_info.mask_token_id), pad_token_id=int(tokenizer_info.pad_token_id))
        model_risk = model_draft = model_types = None
    elif mode == "oracle_refine_mask":
        refine_mask = draft_ids.ne(clean) & valid_mask
        noisy_ids, _, refine_mask = draft_aware_renoise(draft_ids, clean, refine_mask, anchor_mask, int(tokenizer_info.mask_token_id), pad_token_id=int(tokenizer_info.pad_token_id))
        model_risk = model_draft = model_types = None
    else:
        raise ValueError(f"Unknown eval mode={mode!r}")

    timesteps = torch.full((clean.size(0),), max(float((refine_mask & valid_mask).float().mean().item()), 1e-4), device=clean.device)
    kwargs = {}
    if model_draft is not None:
        kwargs = {
            "risk_scores": model_risk,
            "token_confidence": model_draft["token_confidence"],
            "token_entropy": model_draft["token_entropy"],
            "token_margin": model_draft["token_margin"],
            "token_type_ids": model_types,
        }
    sync_if_cuda(clean.device)
    start = now()
    refined_logits = model(noisy_ids, timesteps, **kwargs)["logits"].float()
    sync_if_cuda(clean.device)
    latency = now() - start
    refined_ids = refined_logits.argmax(dim=-1)
    final_logits = gpt2_logits.clone()

    if mode == "draft_refine_adapter_gate":
        final_ids, accepted_mask, _, _ = accept_gate(
            draft_ids,
            refined_logits,
            refined_ids,
            refine_mask,
            draft["token_confidence"],
            risk_scores,
            accept_margin=float(config.get("accept_margin", 0.1)),
            accept_conf_threshold=float(config.get("accept_conf_threshold", 0.6)),
            risk_accept_threshold=float(config.get("risk_accept_threshold", 0.5)),
        )
    else:
        accepted_mask = refine_mask.clone()
        final_ids = draft_ids.clone()
        final_ids[accepted_mask] = refined_ids[accepted_mask]
    changed_mask = accepted_mask & final_ids.ne(draft_ids)
    rows = accepted_mask.nonzero(as_tuple=False)
    if rows.numel() > 0:
        final_logits[rows[:, 0], rows[:, 1]] = refined_logits[rows[:, 0], rows[:, 1]]
    return final_logits, refined_ids, final_ids, changed_mask, accepted_mask, latency


def evaluate(config_path: str, checkpoint: str | None, modes: list[str], eval_steps: int | None = None, save_dir: str | None = None) -> list[dict]:
    config = load_codraft_config(config_path)
    if eval_steps is not None:
        config["eval_steps"] = int(eval_steps)
    if save_dir is not None:
        config["save_dir"] = save_dir
    torch.manual_seed(int(config.get("seed", 7)))
    device = choose_device()
    _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    gpt2_model, _ = load_gpt2(config, tokenizer, device)
    freeze_module(gpt2_model)
    plain_model = None
    adapter_model = None
    if any(mode not in {"gpt2_only", "draft_refine_adapter", "draft_refine_adapter_gate"} for mode in modes):
        plain_model = load_eval_mdlm(config, tokenizer_info, device, None, adapter=False)
        validate_model_surfaces(plain_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))
    if any(mode in {"draft_refine_adapter", "draft_refine_adapter_gate"} for mode in modes):
        adapter_model = load_eval_mdlm(config, tokenizer_info, device, checkpoint, adapter=True)
        validate_model_surfaces(adapter_model, gpt2_model, tokenizer, tokenizer_info, device, int(config["max_length"]))

    accs = {mode: new_acc() for mode in modes}
    for step, batch in enumerate(val_loader):
        if step >= int(config.get("eval_steps", 50)):
            break
        clean = batch.to(device, non_blocking=True)
        valid_mask = clean.ne(int(tokenizer_info.pad_token_id))
        sync_if_cuda(device)
        start = now()
        draft = teacher_forced_gpt2_draft_with_confidence(gpt2_model, clean, int(tokenizer_info.pad_token_id))
        draft["draft_ids"] = draft["draft_ids"].clamp(max=int(tokenizer_info.mask_token_id) - 1)
        sync_if_cuda(device)
        gpt2_latency = now() - start
        risk_scores, low_conf_mask, anchor_mask, token_type_ids = build_risk_batch(config, draft, clean, tokenizer_info)
        comm_mb = estimate_draft_comm_mb(clean.size(0), clean.size(1))
        for mode in modes:
            model = adapter_model if mode in {"draft_refine_adapter", "draft_refine_adapter_gate"} else plain_model
            logits, refined_ids, final_ids, changed_mask, accepted_mask, latency = run_mode_batch(
                mode,
                model,
                clean,
                valid_mask,
                draft,
                risk_scores,
                low_conf_mask,
                anchor_mask,
                token_type_ids,
                tokenizer_info,
                config,
            )
            if mode == "gpt2_only":
                metric_refine_mask = torch.zeros_like(valid_mask)
            elif mode in {"direct_draft_context", "oracle_refine_mask", "random_refine_mask"}:
                metric_refine_mask = accepted_mask
            else:
                metric_refine_mask = low_conf_mask
            add_row_metrics(
                accs[mode],
                logits,
                clean,
                valid_mask,
                draft["draft_ids"],
                refined_ids,
                final_ids,
                metric_refine_mask,
                changed_mask,
                accepted_mask,
                gpt2_latency + latency,
                comm_mb,
            )
        if step == 0 or (step + 1) % int(config.get("log_every", 100)) == 0:
            print(f"eval_step={step + 1}/{int(config.get('eval_steps', 50))}", flush=True)
    rows = [finalize(mode, acc) for mode, acc in accs.items()]
    out_dir = Path(config["save_dir"])
    save_csv(out_dir / "codraft_eval.csv", rows, COLUMNS)
    save_json(out_dir / "codraft_eval.json", {"benchmark": rows, "modes": modes, "checkpoint": checkpoint})
    print(format_table(rows, COLUMNS))
    print(f"saved_codraft_eval={out_dir / 'codraft_eval.csv'}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--modes", default="gpt2_only,direct_draft_context,low_conf_refine,draft_refine_adapter,draft_refine_adapter_gate,random_refine_mask,oracle_refine_mask")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    evaluate(args.config, args.checkpoint, modes, args.eval_steps, args.save_dir)


if __name__ == "__main__":
    main()
