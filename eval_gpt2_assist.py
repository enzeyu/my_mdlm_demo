"""Evaluate AR-Guided Diffusion Verification with GPT-2 candidates and MDLM reranking."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml

from data_real import build_dataloaders, mask_tokens
from metrics import format_table, now, sync_if_cuda
from models_mdlm_wrapper import build_edge_mdlm_model


EVAL_COLUMNS = [
    "mode",
    "loss",
    "perplexity",
    "top1_acc",
    "top5_acc",
    "hard_top1_acc",
    "hard_top5_acc",
    "correction_rate",
    "regression_rate",
    "query_ratio",
    "communication_MB",
    "quality_gain_per_MB",
    "latency",
    "tokens_per_sec",
]

COMPARISON_COLUMNS = [
    "comparison",
    "top1_gain",
    "top5_gain",
    "hard_top1_gain",
    "hard_top5_gain",
    "extra_comm_MB",
    "extra_latency",
    "conclusion",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "edge_model_name_or_path" in config and "pretrained_edge_path" not in config:
        config["pretrained_edge_path"] = config["edge_model_name_or_path"]
    config.setdefault("uncertainty_score", "entropy")
    config.setdefault("hard_token_ratio", 0.3)
    config.setdefault("device_top_k", 20)
    config.setdefault("gpt2_assist_alpha", 0.5)
    config.setdefault("gpt2_query_batch_size", 64)
    return config


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(config: dict, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)
    return Path(config["save_dir"]) / "checkpoint.pt"


def load_device_gpt2(config: dict, mdlm_tokenizer, device: torch.device):
    """Load frozen GPT-2 small and verify token-id compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_files_only = bool(config.get("hf_local_files_only", True))
    configured = str(config.get("device_model_name_or_path", "/mnt/data/enzeyu/hf_downloads/models/gpt2"))
    candidates = [configured]
    if not Path(configured).exists():
        candidates.append("gpt2")

    errors = []
    for name in dict.fromkeys(candidates):
        try:
            gpt2_tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_pretrained(name, local_files_only=local_files_only)
            if gpt2_tokenizer.pad_token is None:
                gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            check_tokenizer_compatibility(mdlm_tokenizer, gpt2_tokenizer, model.config.vocab_size)
            model.requires_grad_(False)
            model.eval()
            return model.to(device), gpt2_tokenizer, name
        except Exception as exc:  # noqa: BLE001 - report every attempted source.
            errors.append(f"{name}: {exc}")
            if Path(configured).exists():
                break
    raise RuntimeError("failed to load compatible GPT-2 candidate generator: " + " | ".join(errors))


def check_tokenizer_compatibility(mdlm_tokenizer, gpt2_tokenizer, gpt2_vocab_size: int) -> None:
    """Ensure candidate ids can be fused into MDLM logits without remapping."""
    mdlm_vocab_size = len(mdlm_tokenizer)
    if gpt2_vocab_size > mdlm_vocab_size:
        raise ValueError(f"GPT-2 vocab_size={gpt2_vocab_size} exceeds MDLM vocab_size={mdlm_vocab_size}")
    if mdlm_tokenizer.eos_token_id != gpt2_tokenizer.eos_token_id:
        raise ValueError(
            f"Tokenizer EOS mismatch: MDLM eos={mdlm_tokenizer.eos_token_id}, "
            f"GPT-2 eos={gpt2_tokenizer.eos_token_id}"
        )
    if mdlm_tokenizer.pad_token_id != gpt2_tokenizer.pad_token_id:
        raise ValueError(
            f"Tokenizer PAD mismatch: MDLM pad={mdlm_tokenizer.pad_token_id}, "
            f"GPT-2 pad={gpt2_tokenizer.pad_token_id}"
        )
    if mdlm_vocab_size != gpt2_vocab_size:
        mask_id = int(mdlm_tokenizer.mask_token_id)
        if mdlm_vocab_size == gpt2_vocab_size + 1 and mask_id == gpt2_vocab_size:
            print(
                "tokenizer_compatibility=ok "
                f"gpt2_vocab={gpt2_vocab_size} mdlm_vocab={mdlm_vocab_size} "
                f"extra_mdlm_mask_id={mask_id}; GPT-2 candidates stay within base vocab"
            )
        else:
            raise ValueError(f"Tokenizer vocab mismatch: GPT-2={gpt2_vocab_size}, MDLM={mdlm_vocab_size}")


def batched(items: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, items.size(0), batch_size):
        yield items[start : start + batch_size]


def select_hard_mask(edge_logits: torch.Tensor, target_mask: torch.Tensor, config: dict) -> torch.Tensor:
    """Select high-uncertainty target_mask positions from edge MDLM logits."""
    hard_mask = torch.zeros_like(target_mask)
    masked_count = int(target_mask.sum().item())
    if masked_count == 0:
        return hard_mask

    masked_logits = edge_logits[target_mask].float()
    probs = torch.softmax(masked_logits, dim=-1)
    score_name = str(config.get("uncertainty_score", "entropy"))
    if score_name == "inverse_confidence":
        scores = 1.0 - probs.max(dim=-1).values
        largest = True
    elif score_name == "margin":
        top2 = probs.topk(2, dim=-1).values
        scores = top2[:, 0] - top2[:, 1]
        largest = False
    else:
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        scores = -(probs * log_probs).sum(dim=-1)
        largest = True

    k = max(1, int(math.ceil(masked_count * float(config.get("hard_token_ratio", 0.3)))))
    selected = scores.topk(min(k, masked_count), largest=largest).indices
    coords = target_mask.nonzero(as_tuple=False)[selected]
    hard_mask[coords[:, 0], coords[:, 1]] = True
    return hard_mask


@torch.no_grad()
def gpt2_candidates_for_positions(
    gpt2_model,
    noised: torch.Tensor,
    positions: torch.Tensor,
    mask_token_id: int,
    eos_token_id: int,
    top_k: int,
    query_batch_size: int,
):
    """Return GPT-2 next-token top-k ids/log-probs for requested positions."""
    if positions.numel() == 0:
        empty_ids = noised.new_empty((0, top_k))
        empty_log_probs = noised.new_empty((0, top_k), dtype=torch.float32)
        return empty_ids, empty_log_probs

    all_ids = []
    all_log_probs = []
    for chunk in batched(positions, query_batch_size):
        lengths = [max(1, int(pos.item())) for _, pos in chunk]
        max_len = max(lengths)
        input_ids = noised.new_full((chunk.size(0), max_len), eos_token_id)
        attention_mask = noised.new_zeros((chunk.size(0), max_len))
        for row_index, (batch_index, pos) in enumerate(chunk.tolist()):
            if pos == 0:
                prefix = noised.new_tensor([eos_token_id])
            else:
                prefix = noised[batch_index, :pos].clone()
                prefix[prefix.eq(mask_token_id)] = eos_token_id
            input_ids[row_index, : prefix.numel()] = prefix
            attention_mask[row_index, : prefix.numel()] = 1

        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        last_indices = torch.tensor(lengths, device=noised.device, dtype=torch.long) - 1
        next_logits = outputs.logits[torch.arange(chunk.size(0), device=noised.device), last_indices].float()
        top = next_logits.topk(min(top_k, next_logits.size(-1)), dim=-1)
        top_log_probs = torch.log_softmax(next_logits, dim=-1).gather(1, top.indices)
        all_ids.append(top.indices)
        all_log_probs.append(top_log_probs)

    return torch.cat(all_ids, dim=0), torch.cat(all_log_probs, dim=0)


@torch.no_grad()
def gpt2_only_batch_metrics(
    gpt2_model,
    noised: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    hard_mask: torch.Tensor,
    mask_token_id: int,
    eos_token_id: int,
    query_batch_size: int,
) -> dict:
    """Evaluate GPT-2 left-context prediction on all masked positions."""
    positions = target_mask.nonzero(as_tuple=False)
    hard_lookup = hard_mask[target_mask]
    totals = new_metric_accumulator()
    if positions.numel() == 0:
        return totals

    hard_offset = 0
    for chunk in batched(positions, query_batch_size):
        lengths = [max(1, int(pos.item())) for _, pos in chunk]
        max_len = max(lengths)
        input_ids = noised.new_full((chunk.size(0), max_len), eos_token_id)
        attention_mask = noised.new_zeros((chunk.size(0), max_len))
        for row_index, (batch_index, pos) in enumerate(chunk.tolist()):
            if pos == 0:
                prefix = noised.new_tensor([eos_token_id])
            else:
                prefix = noised[batch_index, :pos].clone()
                prefix[prefix.eq(mask_token_id)] = eos_token_id
            input_ids[row_index, : prefix.numel()] = prefix
            attention_mask[row_index, : prefix.numel()] = 1

        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        last_indices = torch.tensor(lengths, device=noised.device, dtype=torch.long) - 1
        logits = outputs.logits[torch.arange(chunk.size(0), device=noised.device), last_indices].float()
        chunk_labels = labels[chunk[:, 0], chunk[:, 1]]
        chunk_hard = hard_lookup[hard_offset : hard_offset + chunk.size(0)]
        hard_offset += chunk.size(0)
        add_classification_metrics(totals, logits, chunk_labels, chunk_hard)
    return totals


def new_metric_accumulator() -> dict:
    return {
        "loss_sum": 0.0,
        "top1_sum": 0.0,
        "top5_sum": 0.0,
        "hard_top1_sum": 0.0,
        "hard_top5_sum": 0.0,
        "tokens": 0,
        "hard_tokens": 0,
        "corrected": 0,
        "regressed": 0,
        "edge_wrong": 0,
        "edge_correct": 0,
        "latency": 0.0,
        "communication_MB": 0.0,
        "queries": 0,
        "batches": 0,
    }


def add_classification_metrics(acc: dict, logits: torch.Tensor, labels: torch.Tensor, hard_mask_flat: torch.Tensor) -> None:
    vocab = logits.size(-1)
    token_count = int(labels.numel())
    if token_count == 0:
        return
    loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="sum")
    top1_pred = logits.argmax(dim=-1)
    top5_hit = logits.topk(min(5, vocab), dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1)
    acc["loss_sum"] += float(loss.item())
    acc["top1_sum"] += int(top1_pred.eq(labels).sum().item())
    acc["top5_sum"] += int(top5_hit.sum().item())
    acc["tokens"] += token_count

    if hard_mask_flat.any():
        hard_logits = logits[hard_mask_flat]
        hard_labels = labels[hard_mask_flat]
        hard_top1 = hard_logits.argmax(dim=-1)
        hard_top5 = hard_logits.topk(min(5, vocab), dim=-1).indices.eq(hard_labels.unsqueeze(-1)).any(dim=-1)
        acc["hard_top1_sum"] += int(hard_top1.eq(hard_labels).sum().item())
        acc["hard_top5_sum"] += int(hard_top5.sum().item())
        acc["hard_tokens"] += int(hard_labels.numel())


def add_edge_vs_assist_metrics(
    acc: dict,
    logits: torch.Tensor,
    edge_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    hard_mask: torch.Tensor,
) -> None:
    add_classification_metrics(acc, logits[target_mask], labels[target_mask], hard_mask[target_mask])
    edge_pred = edge_logits[target_mask].argmax(dim=-1)
    assist_pred = logits[target_mask].argmax(dim=-1)
    selected_labels = labels[target_mask]
    edge_correct = edge_pred.eq(selected_labels)
    assist_correct = assist_pred.eq(selected_labels)
    acc["corrected"] += int((~edge_correct & assist_correct).sum().item())
    acc["regressed"] += int((edge_correct & ~assist_correct).sum().item())
    acc["edge_wrong"] += int((~edge_correct).sum().item())
    acc["edge_correct"] += int(edge_correct.sum().item())


def apply_candidate_bias(
    edge_logits: torch.Tensor,
    hard_positions: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if hard_positions.numel() == 0 or alpha == 0.0:
        return edge_logits
    final_logits = edge_logits.clone()
    rows = hard_positions[:, 0]
    cols = hard_positions[:, 1]
    final_logits[rows[:, None], cols[:, None], candidate_ids] += float(alpha) * candidate_log_probs
    return final_logits


def random_candidate_ids(candidate_ids: torch.Tensor, vocab_size: int, mask_token_id: int) -> torch.Tensor:
    if candidate_ids.numel() == 0:
        return candidate_ids
    max_base = vocab_size - 1 if mask_token_id < vocab_size else vocab_size
    ids = torch.randint(0, max_base, candidate_ids.shape, device=candidate_ids.device)
    if mask_token_id < vocab_size:
        ids = torch.where(ids >= mask_token_id, ids + 1, ids)
    return ids.clamp_max(vocab_size - 1)


def communication_mb(num_tokens: int, top_k: int) -> float:
    return num_tokens * top_k * (4 + 4) / (1024 * 1024)


def finalize_rows(accumulators: dict[str, dict], total_masked: int, edge_top5: float) -> list[dict]:
    rows = []
    for mode in ["edge_only", "gpt2_only", "random_assist", "gpt2_assist"]:
        acc = accumulators[mode]
        tokens = max(int(acc["tokens"]), 1)
        hard_tokens = max(int(acc["hard_tokens"]), 1)
        loss = acc["loss_sum"] / tokens
        comm = float(acc["communication_MB"])
        top5 = acc["top5_sum"] / tokens
        rows.append(
            {
                "mode": mode,
                "loss": loss,
                "perplexity": float(math.exp(min(loss, 50.0))),
                "top1_acc": acc["top1_sum"] / tokens,
                "top5_acc": top5,
                "hard_top1_acc": acc["hard_top1_sum"] / hard_tokens,
                "hard_top5_acc": acc["hard_top5_sum"] / hard_tokens,
                "correction_rate": acc["corrected"] / max(acc["edge_wrong"], 1),
                "regression_rate": acc["regressed"] / max(acc["edge_correct"], 1),
                "query_ratio": acc["queries"] / max(total_masked, 1),
                "communication_MB": comm,
                "quality_gain_per_MB": (top5 - edge_top5) / max(comm, 1e-12),
                "latency": acc["latency"] / max(acc["batches"], 1),
                "tokens_per_sec": acc["tokens"] / max(acc["latency"], 1e-12),
            }
        )
    return rows


def comparison_rows(rows: list[dict]) -> list[dict]:
    by_mode = {row["mode"]: row for row in rows}
    assist = by_mode["gpt2_assist"]
    output = []
    for baseline_name in ["edge_only", "random_assist"]:
        baseline = by_mode[baseline_name]
        top1_gain = assist["top1_acc"] - baseline["top1_acc"]
        top5_gain = assist["top5_acc"] - baseline["top5_acc"]
        hard_top1_gain = assist["hard_top1_acc"] - baseline["hard_top1_acc"]
        hard_top5_gain = assist["hard_top5_acc"] - baseline["hard_top5_acc"]
        conclusion = "supports" if top5_gain > 0.0 and hard_top5_gain > 0.0 else "not_clear"
        output.append(
            {
                "comparison": f"gpt2_assist vs {baseline_name}",
                "top1_gain": top1_gain,
                "top5_gain": top5_gain,
                "hard_top1_gain": hard_top1_gain,
                "hard_top5_gain": hard_top5_gain,
                "extra_comm_MB": assist["communication_MB"] - baseline["communication_MB"],
                "extra_latency": assist["latency"] - baseline["latency"],
                "conclusion": conclusion,
            }
        )
    return output


@torch.no_grad()
def evaluate(model, gpt2_model, val_loader, config, tokenizer_info, device) -> tuple[list[dict], list[dict]]:
    model.eval()
    gpt2_model.eval()
    accumulators = {mode: new_metric_accumulator() for mode in ["edge_only", "gpt2_only", "random_assist", "gpt2_assist"]}
    total_masked = 0
    top_k = int(config.get("device_top_k", 20))
    query_batch_size = int(config.get("gpt2_query_batch_size", 64))
    alpha = float(config.get("gpt2_assist_alpha", 0.5))

    for batch in itertools.islice(val_loader, int(config["eval_steps"])):
        clean = batch.to(device, non_blocking=True)
        noised, target_mask = mask_tokens(
            clean,
            tokenizer_info.mask_token_id,
            tokenizer_info.pad_token_id,
            float(config.get("mask_ratio", 0.15)),
        )
        timesteps = torch.full((clean.size(0),), float(config.get("mask_ratio", 0.15)), device=device)
        total_masked += int(target_mask.sum().item())

        sync_if_cuda(device)
        start = now()
        edge_outputs = model(noised, timesteps)
        edge_logits = edge_outputs["logits"]
        sync_if_cuda(device)
        edge_latency = now() - start
        hard_mask = select_hard_mask(edge_logits, target_mask, config)
        hard_positions = hard_mask.nonzero(as_tuple=False)

        edge_acc = accumulators["edge_only"]
        add_edge_vs_assist_metrics(edge_acc, edge_logits, edge_logits, clean, target_mask, hard_mask)
        edge_acc["latency"] += edge_latency
        edge_acc["batches"] += 1

        sync_if_cuda(device)
        gpt2_start = now()
        gpt2_acc_batch = gpt2_only_batch_metrics(
            gpt2_model,
            noised,
            clean,
            target_mask,
            hard_mask,
            int(tokenizer_info.mask_token_id),
            int(tokenizer_info.pad_token_id),
            query_batch_size,
        )
        sync_if_cuda(device)
        gpt2_latency = now() - gpt2_start
        gpt2_acc = accumulators["gpt2_only"]
        for key, value in gpt2_acc_batch.items():
            gpt2_acc[key] += value
        gpt2_acc["queries"] += int(target_mask.sum().item())
        gpt2_acc["communication_MB"] += communication_mb(int(target_mask.sum().item()), top_k)
        gpt2_acc["latency"] += gpt2_latency
        gpt2_acc["batches"] += 1

        sync_if_cuda(device)
        assist_start = now()
        candidate_ids, candidate_log_probs = gpt2_candidates_for_positions(
            gpt2_model,
            noised,
            hard_positions,
            int(tokenizer_info.mask_token_id),
            int(tokenizer_info.pad_token_id),
            top_k,
            query_batch_size,
        )
        assist_logits = apply_candidate_bias(edge_logits, hard_positions, candidate_ids, candidate_log_probs, alpha)
        random_ids = random_candidate_ids(candidate_ids, edge_logits.size(-1), int(tokenizer_info.mask_token_id))
        random_logits = apply_candidate_bias(edge_logits, hard_positions, random_ids, candidate_log_probs, alpha)
        sync_if_cuda(device)
        assist_latency = now() - assist_start

        queried = int(hard_positions.size(0))
        comm = communication_mb(queried, top_k)
        for mode_name, logits in [("random_assist", random_logits), ("gpt2_assist", assist_logits)]:
            acc = accumulators[mode_name]
            add_edge_vs_assist_metrics(acc, logits, edge_logits, clean, target_mask, hard_mask)
            acc["queries"] += queried
            acc["communication_MB"] += comm
            acc["latency"] += edge_latency + assist_latency
            acc["batches"] += 1

    edge_tokens = max(accumulators["edge_only"]["tokens"], 1)
    edge_top5 = accumulators["edge_only"]["top5_sum"] / edge_tokens
    rows = finalize_rows(accumulators, total_masked, edge_top5)
    return rows, comparison_rows(rows)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_skip_outputs(config: dict, reason: str) -> None:
    save_dir = Path(config["save_dir"])
    payload = {
        "status": "skipped",
        "reason": reason,
        "dataset_cache_dir": config.get("dataset_cache_dir", "/mnt/data/enzeyu/hf_downloads/datasets"),
        "model_dir": "/mnt/data/enzeyu/hf_downloads/models",
    }
    save_json(save_dir / "gpt2_assist_eval.json", payload)
    save_csv(save_dir / "gpt2_assist_eval.csv", [], EVAL_COLUMNS)
    lines = [
        "# AR-Guided Diffusion Verification",
        "",
        "Status: skipped",
        "",
        f"Reason: {reason}",
        "",
        f"Expected dataset cache: `{payload['dataset_cache_dir']}`",
        "Please place/download the dataset under that directory, then rerun the same command.",
    ]
    (save_dir / "gpt2_assist_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evaluation_skipped={reason}")
    print(f"saved_skip_summary={save_dir / 'gpt2_assist_summary.md'}")


def write_summary(path: Path, rows: list[dict], comparisons: list[dict], config: dict, gpt2_source: str) -> None:
    by_mode = {row["mode"]: row for row in rows}
    edge = by_mode["edge_only"]
    random = by_mode["random_assist"]
    assist = by_mode["gpt2_assist"]
    assist_vs_edge = comparisons[0]
    assist_vs_random = comparisons[1]
    beats_edge = assist_vs_edge["top5_gain"] > 0 and assist_vs_edge["hard_top5_gain"] > 0 and assist["perplexity"] <= edge["perplexity"]
    beats_random = assist_vs_random["top5_gain"] > 0 and assist_vs_random["hard_top5_gain"] > 0
    hard_main = abs(assist_vs_edge["hard_top5_gain"]) > abs(assist_vs_edge["top5_gain"])
    correction_ok = assist["correction_rate"] > assist["regression_rate"]
    comm_ok = assist["communication_MB"] < 1.0
    hypothesis_ok = beats_edge and beats_random and correction_ok

    lines = [
        "# AR-Guided Diffusion Verification",
        "",
        f"- Edge MDLM: `{config.get('pretrained_edge_path', config.get('edge_model_name_or_path'))}`",
        f"- Device GPT-2: `{gpt2_source}`",
        f"- Uncertainty score: `{config.get('uncertainty_score', 'entropy')}`",
        f"- Hard token ratio: `{config.get('hard_token_ratio', 0.3)}`",
        f"- Device top-k: `{config.get('device_top_k', 20)}`",
        f"- Assist alpha: `{config.get('gpt2_assist_alpha', 0.5)}`",
        "",
        "| Mode | Loss | PPL | Top1 | Top5 | Hard Top1 | Hard Top5 | Correction | Regression | Query Ratio | Comm MB | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['loss']:.4f} | {row['perplexity']:.4f} | "
            f"{row['top1_acc']:.4f} | {row['top5_acc']:.4f} | "
            f"{row['hard_top1_acc']:.4f} | {row['hard_top5_acc']:.4f} | "
            f"{row['correction_rate']:.4f} | {row['regression_rate']:.4f} | "
            f"{row['query_ratio']:.4f} | {row['communication_MB']:.6f} | {row['latency']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Top1 Gain | Top5 Gain | Hard Top1 Gain | Hard Top5 Gain | Extra Comm MB | Extra Latency | Conclusion |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {row['top1_gain']:.6f} | {row['top5_gain']:.6f} | "
            f"{row['hard_top1_gain']:.6f} | {row['hard_top5_gain']:.6f} | "
            f"{row['extra_comm_MB']:.6f} | {row['extra_latency']:.6f} | {row['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"1. gpt2_assist 是否优于 edge_only？{'是' if beats_edge else '否，当前指标不足以支持'}。Top5 gain={assist_vs_edge['top5_gain']:.6f}，Hard Top5 gain={assist_vs_edge['hard_top5_gain']:.6f}，PPL delta={assist['perplexity'] - edge['perplexity']:.6f}。",
            f"2. gpt2_assist 是否优于 random_assist？{'是' if beats_random else '否'}。Top5 gain={assist_vs_random['top5_gain']:.6f}，Hard Top5 gain={assist_vs_random['hard_top5_gain']:.6f}。",
            f"3. 提升是否主要发生在 hard tokens 上？{'是' if hard_main else '否'}。",
            f"4. correction_rate 是否高于 regression_rate？{'是' if correction_ok else '否'}。correction={assist['correction_rate']:.6f}，regression={assist['regression_rate']:.6f}。",
            f"5. 通信开销是否可接受？{'是' if comm_ok else '需要结合系统预算判断'}。当前 assist 通信量约 {assist['communication_MB']:.6f} MB。",
            f"6. 是否支持“端侧 AR 小模型辅助边侧扩散模型”的研究假设？{'支持' if hypothesis_ok else '暂不支持，证据还不充分'}。",
            "7. 如果效果不明显，可能原因包括：GPT-2 只能看左上下文而 MDLM 使用双向 masked context；GPT-2 top-k 未覆盖真实 token；log-prob bias 的 alpha 未调优；候选只改变 logits 不改变 MDLM hidden state；WikiText masked token 中存在需要右上下文的歧义；GPT-2 与 MDLM 训练目标和数据分布不同。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_tables(rows: list[dict], comparisons: list[dict]) -> None:
    display_rows = [
        {
            "Mode": row["mode"],
            "Loss": row["loss"],
            "PPL": row["perplexity"],
            "Top1": row["top1_acc"],
            "Top5": row["top5_acc"],
            "Hard Top1": row["hard_top1_acc"],
            "Hard Top5": row["hard_top5_acc"],
            "Correction": row["correction_rate"],
            "Regression": row["regression_rate"],
            "Query Ratio": row["query_ratio"],
            "Comm MB": row["communication_MB"],
            "Latency": row["latency"],
        }
        for row in rows
    ]
    print(
        format_table(
            display_rows,
            ["Mode", "Loss", "PPL", "Top1", "Top5", "Hard Top1", "Hard Top5", "Correction", "Regression", "Query Ratio", "Comm MB", "Latency"],
        )
    )
    print()
    display_comparisons = [
        {
            "Comparison": row["comparison"],
            "Top1 Gain": row["top1_gain"],
            "Top5 Gain": row["top5_gain"],
            "Hard Top1 Gain": row["hard_top1_gain"],
            "Hard Top5 Gain": row["hard_top5_gain"],
            "Extra Comm MB": row["extra_comm_MB"],
            "Extra Latency": row["extra_latency"],
            "Conclusion": row["conclusion"],
        }
        for row in comparisons
    ]
    print(
        format_table(
            display_comparisons,
            ["Comparison", "Top1 Gain", "Top5 Gain", "Hard Top1 Gain", "Hard Top5 Gain", "Extra Comm MB", "Extra Latency", "Conclusion"],
        )
    )


def run_from_args(config_path: str, ckpt_path: str | None = None) -> tuple[list[dict], list[dict]]:
    config = load_config(config_path)
    device = choose_device()
    try:
        _, val_loader, tokenizer, tokenizer_info = build_dataloaders(config)
    except Exception as exc:  # noqa: BLE001 - make missing local datasets explicit.
        reason = (
            f"failed to load dataset `{config.get('dataset_name', 'wikitext-2')}` from "
            f"`{config.get('dataset_cache_dir', '/mnt/data/enzeyu/hf_downloads/datasets')}`: {exc}"
        )
        write_skip_outputs(config, reason)
        return [], []
    model = build_edge_mdlm_model(config, tokenizer_info.vocab_size, tokenizer_info.pad_token_id, tokenizer_info.mask_token_id).to(device)
    resolved_ckpt = resolve_checkpoint(config, ckpt_path)
    if resolved_ckpt.exists():
        checkpoint = torch.load(resolved_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing or unexpected:
            print(f"checkpoint_load_warning missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"checkpoint_load_skipped missing_path={resolved_ckpt}")
    gpt2_model, _, gpt2_source = load_device_gpt2(config, tokenizer, device)
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")
    print(f"device_gpt2_source={gpt2_source}")

    rows, comparisons = evaluate(model, gpt2_model, val_loader, config, tokenizer_info, device)
    save_dir = Path(config["save_dir"])
    save_csv(save_dir / "gpt2_assist_eval.csv", rows, EVAL_COLUMNS)
    save_json(save_dir / "gpt2_assist_eval.json", {"benchmark": rows, "comparisons": comparisons})
    write_summary(save_dir / "gpt2_assist_summary.md", rows, comparisons, config, gpt2_source)
    print_tables(rows, comparisons)
    print(f"saved_gpt2_assist={save_dir / 'gpt2_assist_eval.csv'}")
    print(f"saved_gpt2_assist_summary={save_dir / 'gpt2_assist_summary.md'}")
    return rows, comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    run_from_args(args.config, args.ckpt)


if __name__ == "__main__":
    main()
