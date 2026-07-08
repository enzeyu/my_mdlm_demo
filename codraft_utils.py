"""Utilities for CoDraft-Diff AR-to-MDLM collaborative refinement."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


def flatten_codraft_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested CoDraft YAML while preserving existing flat keys."""
    cfg = dict(config)
    dataset = cfg.get("dataset", cfg.get("dataset_name", "wikitext2"))
    cfg.setdefault("dataset_name", dataset)
    cfg.setdefault("tokenizer_name", cfg.get("tokenizer_name", cfg.get("device_model_name_or_path")))

    device_ar = cfg.get("device_ar", {}) or {}
    edge_mdlm = cfg.get("edge_mdlm", {}) or {}
    draft = cfg.get("draft", {}) or {}
    risk = cfg.get("risk", {}) or {}
    gate = cfg.get("accept_gate", {}) or {}
    training = cfg.get("training", {}) or {}
    weights = cfg.get("loss_weights", {}) or {}

    if device_ar.get("model_name_or_path"):
        cfg.setdefault("device_model_name_or_path", device_ar["model_name_or_path"])
    cfg.setdefault("device_model_name_or_path", cfg.get("device_model_name_or_path", "/mnt/data/enzeyu/hf_downloads/models/gpt2"))
    cfg.setdefault("tokenizer_name", cfg["device_model_name_or_path"])
    if edge_mdlm.get("model_name_or_path"):
        cfg.setdefault("pretrained_edge_path", edge_mdlm["model_name_or_path"])
        cfg.setdefault("edge_model_name_or_path", edge_mdlm["model_name_or_path"])
        cfg.setdefault("use_pretrained_edge", True)

    for nested_key, flat_key in [
        ("hidden_size", "edge_hidden_size"),
        ("num_layers", "edge_layers"),
        ("num_heads", "edge_heads"),
        ("dropout", "dropout"),
        ("use_draft_conditioning", "use_draft_conditioning"),
        ("use_risk_embedding", "use_risk_embedding"),
        ("use_confidence_embedding", "use_confidence_embedding"),
        ("use_token_type_embedding", "use_token_type_embedding"),
        ("use_draft_adapter", "use_draft_adapter"),
        ("adapter_bottleneck", "adapter_bottleneck"),
        ("freeze_mdlm_backbone", "freeze_mdlm_backbone"),
    ]:
        if nested_key in edge_mdlm:
            cfg.setdefault(flat_key, edge_mdlm[nested_key])

    for nested_key in ["max_new_tokens", "temperature", "top_k"]:
        if nested_key in draft:
            cfg.setdefault(nested_key, draft[nested_key])
    for nested_key in ["confidence_threshold", "entropy_threshold", "margin_threshold", "expand_window", "min_span_len"]:
        if nested_key in risk:
            cfg.setdefault(nested_key, risk[nested_key])
    for nested_key in ["accept_margin", "accept_conf_threshold", "risk_accept_threshold"]:
        if nested_key in gate:
            cfg.setdefault(nested_key, gate[nested_key])
    for nested_key in ["batch_size", "lr", "adapter_lr", "max_steps", "log_every", "eval_every", "save_every"]:
        if nested_key in training:
            flat_key = "train_steps" if nested_key == "max_steps" else nested_key
            cfg.setdefault(flat_key, training[nested_key])
    for nested_key, flat_key in [
        ("draft_refine_loss", "draft_refine_loss_weight"),
        ("confidence_calibration", "confidence_calibration_weight"),
        ("refine_ratio_budget", "refine_ratio_budget_weight"),
    ]:
        if nested_key in weights:
            cfg.setdefault(flat_key, weights[nested_key])

    cfg.setdefault("save_dir", f"results/{cfg.get('experiment_name', 'codraft_diff')}")
    cfg.setdefault("hf_local_files_only", True)
    cfg.setdefault("max_length", 128)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("train_steps", 1000)
    cfg.setdefault("eval_steps", 50)
    cfg.setdefault("eval_batches", 10)
    cfg.setdefault("mask_ratio", 0.15)
    cfg.setdefault("refine_ratio", 0.2)
    cfg.setdefault("confidence_threshold", 0.5)
    cfg.setdefault("entropy_threshold", 3.0)
    cfg.setdefault("margin_threshold", 0.1)
    cfg.setdefault("expand_window", 1)
    cfg.setdefault("min_span_len", 1)
    cfg.setdefault("accept_margin", 0.1)
    cfg.setdefault("accept_conf_threshold", 0.6)
    cfg.setdefault("risk_accept_threshold", 0.5)
    return cfg


def load_codraft_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return flatten_codraft_config(yaml.safe_load(handle))


@torch.no_grad()
def generate_gpt2_draft_with_confidence(
    gpt2_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> dict[str, torch.Tensor]:
    """Autoregressively generate a GPT-2 draft and token-level confidence stats."""
    del tokenizer
    gpt2_model.eval()
    generated = prompt_ids.clone()
    logits_rows = []
    conf_rows = []
    entropy_rows = []
    margin_rows = []
    logprob_rows = []
    for _ in range(int(max_new_tokens)):
        outputs = gpt2_model(input_ids=generated, attention_mask=torch.ones_like(generated))
        logits = outputs.logits[:, -1, :].float() / max(float(temperature), 1e-6)
        if top_k and int(top_k) > 0 and int(top_k) < logits.size(-1):
            keep = logits.topk(int(top_k), dim=-1)
            filtered = logits.new_full(logits.shape, -1e4)
            filtered.scatter_(1, keep.indices, keep.values)
            logits = filtered
        probs = torch.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)
        log_probs = torch.log_softmax(logits, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        logits_rows.append(logits)
        conf_rows.append(top2[:, 0])
        entropy_rows.append(-(probs * log_probs).sum(dim=-1))
        margin_rows.append(top2[:, 0] - top2[:, 1])
        logprob_rows.append(log_probs.gather(1, next_ids).squeeze(1))
        generated = torch.cat([generated, next_ids], dim=1)
    return {
        "draft_ids": generated,
        "draft_logits": torch.stack(logits_rows, dim=1) if logits_rows else prompt_ids.new_zeros((*prompt_ids.shape, 0)),
        "token_confidence": torch.stack(conf_rows, dim=1),
        "token_entropy": torch.stack(entropy_rows, dim=1),
        "token_margin": torch.stack(margin_rows, dim=1),
        "token_logprob": torch.stack(logprob_rows, dim=1),
    }


@torch.no_grad()
def teacher_forced_gpt2_draft_with_confidence(gpt2_model, labels: torch.Tensor, bos_token_id: int) -> dict[str, torch.Tensor]:
    """Build length-aligned GPT-2 draft stats for fixed-block training/evaluation."""
    shifted = labels.new_empty(labels.shape)
    shifted[:, 0] = int(bos_token_id)
    shifted[:, 1:] = labels[:, :-1]
    logits = gpt2_model(input_ids=shifted, attention_mask=torch.ones_like(shifted)).logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1)
    draft_ids = top2.indices[..., 0]
    token_confidence = top2.values[..., 0]
    token_entropy = -(probs * log_probs).sum(dim=-1)
    token_margin = top2.values[..., 0] - top2.values[..., 1]
    safe_labels = labels.clamp(max=logits.size(-1) - 1)
    token_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return {
        "draft_ids": draft_ids,
        "draft_logits": logits,
        "token_confidence": token_confidence,
        "token_entropy": token_entropy,
        "token_margin": token_margin,
        "token_logprob": token_logprob,
    }


def estimate_draft_risk(
    draft_ids: torch.Tensor,
    token_confidence: torch.Tensor,
    token_entropy: torch.Tensor,
    token_margin: torch.Tensor,
    confidence_threshold: float,
    entropy_threshold: float,
    margin_threshold: float,
    expand_window: int = 1,
    min_span_len: int = 1,
    pad_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate risky GPT-2 draft positions and stable anchors."""
    valid_mask = torch.ones_like(draft_ids, dtype=torch.bool)
    if pad_token_id is not None:
        valid_mask = draft_ids.ne(int(pad_token_id))
    conf_risk = (float(confidence_threshold) - token_confidence.float()).clamp_min(0.0) / max(float(confidence_threshold), 1e-6)
    entropy_risk = (token_entropy.float() - float(entropy_threshold)).clamp_min(0.0) / max(float(entropy_threshold), 1e-6)
    margin_risk = (float(margin_threshold) - token_margin.float()).clamp_min(0.0) / max(float(margin_threshold), 1e-6)
    risk_scores = torch.stack([conf_risk, entropy_risk, margin_risk], dim=0).amax(dim=0).clamp(0.0, 1.0)
    refine_mask = (
        token_confidence.lt(float(confidence_threshold))
        | token_entropy.gt(float(entropy_threshold))
        | token_margin.lt(float(margin_threshold))
    ) & valid_mask
    if expand_window > 0 and bool(refine_mask.any()):
        expanded = refine_mask.clone()
        for offset in range(1, int(expand_window) + 1):
            expanded[:, offset:] |= refine_mask[:, :-offset]
            expanded[:, :-offset] |= refine_mask[:, offset:]
        refine_mask = expanded & valid_mask
    if int(min_span_len) > 1:
        refine_mask = _drop_short_spans(refine_mask, valid_mask, int(min_span_len))
    anchor_mask = (
        token_confidence.ge(float(confidence_threshold))
        & token_margin.ge(float(margin_threshold))
        & ~refine_mask
        & valid_mask
    )
    return risk_scores.masked_fill(~valid_mask, 0.0), refine_mask, anchor_mask


def _drop_short_spans(refine_mask: torch.Tensor, valid_mask: torch.Tensor, min_span_len: int) -> torch.Tensor:
    kept = torch.zeros_like(refine_mask)
    for row in range(refine_mask.size(0)):
        start = None
        for idx in range(refine_mask.size(1) + 1):
            active = idx < refine_mask.size(1) and bool(refine_mask[row, idx] and valid_mask[row, idx])
            if active and start is None:
                start = idx
            if (not active or idx == refine_mask.size(1)) and start is not None:
                if idx - start >= min_span_len:
                    kept[row, start:idx] = True
                start = None
    return kept & valid_mask


def draft_aware_renoise(
    draft_ids: torch.Tensor,
    labels: torch.Tensor,
    refine_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_token_id: int,
    suspicious_keep_prob: float = 0.0,
    pad_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert GPT-2 draft corruption into MDLM absorbing-mask input."""
    del anchor_mask
    valid_mask = labels.ne(int(pad_token_id)) if pad_token_id is not None else torch.ones_like(labels, dtype=torch.bool)
    target_mask = refine_mask & valid_mask
    if suspicious_keep_prob > 0:
        keep = torch.rand_like(target_mask.float()).lt(float(suspicious_keep_prob))
        target_mask = target_mask & ~keep
    noisy_ids = draft_ids.clone()
    noisy_ids[target_mask] = int(mask_token_id)
    target_labels = labels.clone()
    return noisy_ids, target_labels, target_mask


def cap_refine_mask_by_ratio(
    refine_mask: torch.Tensor,
    risk_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    max_ratio: float,
) -> torch.Tensor:
    """Keep only the highest-risk refine positions under a per-sequence budget."""
    ratio = float(max_ratio)
    if ratio <= 0 or ratio >= 1:
        return refine_mask & valid_mask
    capped = torch.zeros_like(refine_mask)
    for row in range(refine_mask.size(0)):
        candidates = torch.where(refine_mask[row] & valid_mask[row])[0]
        valid_count = int(valid_mask[row].sum().item())
        if candidates.numel() == 0 or valid_count == 0:
            continue
        k = min(candidates.numel(), max(1, int(math.ceil(valid_count * ratio))))
        chosen_local = risk_scores[row, candidates].topk(k, largest=True).indices
        capped[row, candidates[chosen_local]] = True
    return capped & valid_mask


def select_top_ratio_by_score(score: torch.Tensor, valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Select top-scoring valid positions per sequence under a ratio budget."""
    selected = torch.zeros_like(valid_mask)
    ratio = float(ratio)
    if ratio <= 0:
        return selected
    for row in range(valid_mask.size(0)):
        coords = torch.where(valid_mask[row])[0]
        valid_count = int(coords.numel())
        if valid_count == 0:
            continue
        k = min(valid_count, max(1, int(math.ceil(valid_count * ratio))))
        chosen = score[row, coords].float().topk(k, largest=True).indices
        selected[row, coords[chosen]] = True
    return selected & valid_mask


def transition_counts(
    draft_ids: torch.Tensor,
    final_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, int]:
    """Count token-level edit outcome transitions."""
    draft_correct = draft_ids.eq(labels) & valid_mask
    draft_wrong = ~draft_ids.eq(labels) & valid_mask
    final_correct = final_ids.eq(labels) & valid_mask
    return {
        "correct_to_correct": int((draft_correct & final_correct).sum().item()),
        "correct_to_wrong": int((draft_correct & ~final_correct).sum().item()),
        "wrong_to_correct": int((draft_wrong & final_correct).sum().item()),
        "wrong_to_wrong": int((draft_wrong & ~final_correct).sum().item()),
    }


def normalize_feature(feature: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a feature over valid positions in each sequence."""
    out = torch.zeros_like(feature.float())
    for row in range(feature.size(0)):
        mask = valid_mask[row]
        if not bool(mask.any()):
            continue
        vals = feature[row, mask].float()
        low = vals.min()
        high = vals.max()
        out[row, mask] = (vals - low) / (high - low).clamp_min(1e-6)
    return out


def error_aware_score(
    token_confidence: torch.Tensor,
    token_entropy: torch.Tensor,
    token_margin: torch.Tensor,
    valid_mask: torch.Tensor,
    disagreement: torch.Tensor | None = None,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Non-trained draft error score from normalized uncertainty features."""
    inv_conf = normalize_feature(1.0 - token_confidence.float(), valid_mask)
    entropy = normalize_feature(token_entropy.float(), valid_mask)
    inv_margin = normalize_feature(1.0 - token_margin.float(), valid_mask)
    if disagreement is None:
        disagreement = torch.zeros_like(inv_conf)
    else:
        disagreement = normalize_feature(disagreement.float(), valid_mask)
    w1, w2, w3, w4 = [float(value) for value in weights]
    score = w1 * inv_conf + w2 * entropy + w3 * inv_margin + w4 * disagreement
    return score.masked_fill(~valid_mask, float("-inf"))


def utility_accept_gate(
    draft_ids: torch.Tensor,
    refined_logits: torch.Tensor,
    refined_ids: torch.Tensor,
    refine_mask: torch.Tensor,
    token_confidence: torch.Tensor,
    utility_lambda: float = 2.0,
    utility_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Conservative utility-aware gate for accepting edits."""
    probs = torch.softmax(refined_logits.float(), dim=-1)
    refined_prob = probs.gather(-1, refined_ids.unsqueeze(-1)).squeeze(-1)
    safe_draft = draft_ids.clamp(max=refined_logits.size(-1) - 1)
    draft_prob = probs.gather(-1, safe_draft.unsqueeze(-1)).squeeze(-1)
    margin = refined_prob - draft_prob
    disagreement = refined_ids.ne(draft_ids).float()
    high_draft_conf_penalty = token_confidence.float()
    utility = margin + 0.5 * disagreement - float(utility_lambda) * high_draft_conf_penalty * disagreement
    accepted_mask = refine_mask & utility.gt(float(utility_threshold)) & refined_ids.ne(draft_ids)
    final_ids = draft_ids.clone()
    final_ids[accepted_mask] = refined_ids[accepted_mask]
    num_accepted = int(accepted_mask.sum().item())
    accept_rate = num_accepted / max(int(refine_mask.sum().item()), 1)
    return final_ids, accepted_mask, float(accept_rate), num_accepted


def build_token_type_ids(refine_mask: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
    token_type_ids = torch.zeros_like(refine_mask, dtype=torch.long)
    token_type_ids[anchor_mask] = 1
    token_type_ids[refine_mask] = 2
    return token_type_ids


def accept_gate(
    draft_ids: torch.Tensor,
    refined_logits: torch.Tensor,
    refined_ids: torch.Tensor,
    refine_mask: torch.Tensor,
    token_confidence: torch.Tensor,
    risk_scores: torch.Tensor,
    accept_margin: float = 0.1,
    accept_conf_threshold: float = 0.6,
    risk_accept_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Heuristic accept gate for selective MDLM refinements."""
    del token_confidence
    probs = torch.softmax(refined_logits.float(), dim=-1)
    refined_prob = probs.gather(-1, refined_ids.unsqueeze(-1)).squeeze(-1)
    safe_draft = draft_ids.clamp(max=refined_logits.size(-1) - 1)
    draft_prob = probs.gather(-1, safe_draft.unsqueeze(-1)).squeeze(-1)
    accepted_mask = refine_mask & (
        (refined_prob - draft_prob).gt(float(accept_margin))
        | (refined_prob.gt(float(accept_conf_threshold)) & risk_scores.gt(float(risk_accept_threshold)))
    )
    final_ids = draft_ids.clone()
    final_ids[accepted_mask] = refined_ids[accepted_mask]
    num_accepted = int(accepted_mask.sum().item())
    accept_rate = num_accepted / max(int(refine_mask.sum().item()), 1)
    return final_ids, accepted_mask, float(accept_rate), num_accepted


def estimate_draft_comm_mb(
    batch_size: int,
    seq_len: int,
    send_token_ids: bool = True,
    send_confidence: bool = True,
    send_entropy: bool = True,
    send_margin: bool = True,
    dtype_bytes: int = 2,
    send_risk: bool = True,
) -> float:
    bytes_total = 0
    if send_token_ids:
        bytes_total += int(batch_size) * int(seq_len) * 4
    for enabled in [send_confidence, send_entropy, send_margin, send_risk]:
        if enabled:
            bytes_total += int(batch_size) * int(seq_len) * int(dtype_bytes)
    return bytes_total / (1024.0 * 1024.0)


def now() -> float:
    return time.perf_counter()


def save_json(path: str | Path, payload: Any) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row}) if rows else ["mode"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def masked_lm_stats(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float | torch.Tensor | int]:
    if not bool(mask.any()):
        zero = logits.new_tensor(0.0)
        return {"loss": zero, "top1": 0.0, "top5": 0.0, "tokens": 0}
    selected_logits = logits[mask]
    selected_labels = labels[mask].clamp(max=logits.size(-1) - 1)
    loss = F.cross_entropy(selected_logits, selected_labels)
    pred = selected_logits.argmax(dim=-1)
    top5 = selected_logits.topk(min(5, logits.size(-1)), dim=-1).indices.eq(selected_labels.unsqueeze(-1)).any(dim=-1)
    return {
        "loss": loss,
        "top1": float(pred.eq(selected_labels).float().mean().item()),
        "top5": float(top5.float().mean().item()),
        "tokens": int(selected_labels.numel()),
    }


def correction_metrics(
    draft_ids: torch.Tensor,
    refined_ids: torch.Tensor,
    final_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    changed_mask: torch.Tensor,
    refine_mask: torch.Tensor,
) -> dict[str, float]:
    draft_correct = draft_ids.eq(labels) & valid_mask
    draft_wrong = ~draft_ids.eq(labels) & valid_mask
    final_correct = final_ids.eq(labels) & valid_mask
    refined_correct = refined_ids.eq(labels) & refine_mask & valid_mask
    changed = changed_mask & valid_mask
    good_changes = changed & draft_wrong & final_correct
    bad_changes = changed & draft_correct & ~final_correct
    return {
        "draft_token_acc": float(draft_correct.sum().item() / max(int(valid_mask.sum().item()), 1)),
        "refined_token_acc": float(refined_correct.sum().item() / max(int((refine_mask & valid_mask).sum().item()), 1)),
        "final_token_acc": float(final_correct.sum().item() / max(int(valid_mask.sum().item()), 1)),
        "correction_precision": float(good_changes.sum().item() / max(int(changed.sum().item()), 1)),
        "correction_recall": float(good_changes.sum().item() / max(int(draft_wrong.sum().item()), 1)),
        "wrong_edit_rate": float(bad_changes.sum().item() / max(int(draft_correct.sum().item()), 1)),
        "correct_token_preservation": float((draft_correct & final_correct).sum().item() / max(int(draft_correct.sum().item()), 1)),
        "refine_ratio": float((refine_mask & valid_mask).sum().item() / max(int(valid_mask.sum().item()), 1)),
    }
