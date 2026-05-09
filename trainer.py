"""Training orchestration for coarse-to-fine edge-device diffusion LM demos."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from collaboration import CoarseToFineCollaboration
from data import build_datasets
from diffusion_utils import (
    corrupt_tokens,
    cross_entropy_and_grad,
    distinct_and_repetition,
    iterative_sample,
    perplexity_surrogate,
    sample_noise_prob,
    token_recovery_accuracy,
)
from losses import kl_distillation_loss
from metrics import load_results, upsert_result, write_results
from models import build_components


MODES = {"device_only", "edge_only", "vanilla_collaborative_distillation", "coarse_to_fine", "collaborative"}


class ExperimentTrainer:
    def __init__(self, config: dict, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}; expected one of {sorted(MODES)}")
        self.config = config
        self.mode = "vanilla_collaborative_distillation" if mode == "collaborative" else mode
        self.rng = np.random.default_rng(int(config.get("seed", 7)))
        self.train_data, self.val_data, self.tokenizer = build_datasets(config)
        self.coarse_space, self.device_model, self.edge_model = build_components(config, self.tokenizer)
        self.collab = CoarseToFineCollaboration()

    def train(self, persist: bool = True) -> Dict[str, float]:
        steps = int(self.config["training"]["steps"])
        batch_size = int(self.config["data"]["batch_size"])
        train_start = time.perf_counter()
        loss_totals = {"device": 0.0, "edge": 0.0, "align": 0.0, "distill": 0.0}
        acc_total = 0.0

        for step in range(1, steps + 1):
            clean = self._sample_batch(self.train_data, batch_size)
            t = sample_noise_prob(clean.shape[0], self.config, self.rng)
            corrupted, mask = corrupt_tokens(clean, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, t, self.rng)
            clean_coarse, _ = self.coarse_space.encode(clean)
            noisy_coarse, _ = self.coarse_space.encode(corrupted)

            if self.mode == "device_only":
                loss_totals["device"] += self.device_model.train_step(noisy_coarse, clean_coarse, t)

            elif self.mode == "edge_only":
                edge_loss, edge_acc, _, _ = self.edge_model.train_step(corrupted, clean, mask, t, coarse=None)
                loss_totals["edge"] += edge_loss
                acc_total += edge_acc

            elif self.mode == "vanilla_collaborative_distillation":
                edge_loss, edge_acc, teacher_logits, _ = self.edge_model.train_step(corrupted, clean, mask, t, coarse=None)
                device_pred = self.device_model.forward(noisy_coarse, t)
                student_logits = self.coarse_space.coarse_to_token_logits(device_pred, clean.shape[1])
                distill = kl_distillation_loss(student_logits, teacher_logits, mask, self.config["collaboration"].get("temperature", 2.0))
                loss_totals["device"] += self.device_model.train_step(noisy_coarse, clean_coarse, t)
                loss_totals["edge"] += edge_loss
                loss_totals["distill"] += distill
                selected = min(int(mask.sum()), int(self.config["collaboration"].get("top_k_tokens", 16)))
                self.collab.record_distillation(selected, self.tokenizer.vocab_size)
                acc_total += edge_acc

            else:
                device_loss = self.device_model.train_step(noisy_coarse, clean_coarse, t)
                pred_coarse = self.device_model.forward(noisy_coarse, t)
                self.collab.record_coarse(clean.shape[0], pred_coarse.shape[1], pred_coarse.shape[2])
                self.collab.record_hidden_baseline(clean.shape[0], clean.shape[1], int(self.config["edge_model"]["hidden_size"]))
                edge_loss, edge_acc, _, _ = self.edge_model.train_step(corrupted, clean, mask, t, coarse=pred_coarse)
                align = self.coarse_space.alignment_loss(pred_coarse, clean_coarse)
                loss_totals["device"] += device_loss
                loss_totals["edge"] += edge_loss
                loss_totals["align"] += align
                acc_total += edge_acc

        train_time = time.perf_counter() - train_start
        val = self.evaluate()
        sampling = self.measure_generation()
        result = self._result_row(train_time, steps, loss_totals, acc_total / max(steps, 1), val, sampling)
        if persist:
            output_dir = self.config["outputs"]["dir"]
            results = upsert_result(load_results(output_dir), result)
            write_results(results, output_dir)
            self._save_checkpoint_metadata(result)
        return result

    def evaluate(self) -> Dict[str, float]:
        batch_size = int(self.config["data"]["batch_size"])
        max_batches = int(self.config["training"].get("eval_batches", 4))
        total_loss = 0.0
        total_acc = 0.0
        total_align = 0.0
        count = 0
        for start in range(0, min(len(self.val_data), max_batches * batch_size), batch_size):
            clean = self.val_data[start : start + batch_size]
            if len(clean) == 0:
                continue
            t = sample_noise_prob(clean.shape[0], self.config, self.rng)
            corrupted, mask = corrupt_tokens(clean, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, t, self.rng)
            clean_coarse, _ = self.coarse_space.encode(clean)
            noisy_coarse, _ = self.coarse_space.encode(corrupted)
            pred_coarse = self.device_model.forward(noisy_coarse, t)
            if self.mode == "device_only":
                logits = self.coarse_space.coarse_to_token_logits(pred_coarse, clean.shape[1])
            elif self.mode == "edge_only":
                logits = self.edge_model.forward(corrupted, t, coarse=None)
            else:
                logits = self.edge_model.forward(corrupted, t, coarse=pred_coarse)
            loss, _, acc = cross_entropy_and_grad(logits, clean, mask)
            total_loss += loss
            total_acc += acc if acc else token_recovery_accuracy(logits, clean, mask)
            total_align += self.coarse_space.alignment_loss(pred_coarse, clean_coarse)
            count += 1
        val_loss = total_loss / max(count, 1)
        return {
            "val_loss": val_loss,
            "token_acc": total_acc / max(count, 1),
            "masked_token_accuracy": total_acc / max(count, 1),
            "perplexity_surrogate": perplexity_surrogate(val_loss),
            "alignment_loss": total_align / max(count, 1),
        }

    def measure_generation(self) -> Dict[str, float]:
        seq_len = int(self.config["data"]["seq_len"])
        device_steps = int(self.config["diffusion"].get("device_steps", self.config["diffusion"].get("sampling_steps", 4)))
        edge_steps = int(self.config["diffusion"].get("edge_steps", self.config["diffusion"].get("sampling_steps", 4)))

        def device_forward(x, t):
            noisy, _ = self.coarse_space.encode(x)
            pred = self.device_model.forward(noisy, t)
            return self.coarse_space.coarse_to_token_logits(pred, x.shape[1])

        def edge_forward(x, t):
            return self.edge_model.forward(x, t, coarse=None)

        def c2f_forward(x, t):
            noisy, _ = self.coarse_space.encode(x)
            pred = self.device_model.forward(noisy, t)
            return self.edge_model.forward(x, t, coarse=pred)

        _, device_time = iterative_sample(device_forward, seq_len, self.tokenizer.mask_token_id, device_steps, self.rng)
        _, edge_time = iterative_sample(edge_forward, seq_len, self.tokenizer.mask_token_id, edge_steps, self.rng)
        sample_ids, c2f_time = iterative_sample(c2f_forward, seq_len, self.tokenizer.mask_token_id, device_steps + edge_steps, self.rng)

        selected_time = {"device_only": device_time, "edge_only": edge_time}.get(self.mode, c2f_time)
        quality = distinct_and_repetition(sample_ids)
        return {
            "sampling_latency": selected_time,
            "tokens_per_sec": seq_len / max(selected_time, 1e-9),
            "denoising_steps": device_steps + edge_steps if self.mode == "coarse_to_fine" else (edge_steps if self.mode == "edge_only" else device_steps),
            "device_only_generation_time": device_time,
            "edge_only_generation_time": edge_time,
            "coarse_to_fine_generation_time": c2f_time,
            **quality,
        }

    def _sample_batch(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        idx = self.rng.integers(0, len(data), size=(batch_size,))
        return data[idx].copy()

    def _result_row(self, train_time: float, steps: int, losses: dict, train_acc: float, val: dict, sampling: dict) -> Dict[str, float]:
        seq_len = int(self.config["data"]["seq_len"])
        coarse_len = seq_len if self.config["coarse_space"].get("compression_method", "linear") != "pooling" else int(
            np.ceil(seq_len / int(self.config["coarse_space"].get("segment_size", 2)))
        )
        coarse_dim = int(self.config["coarse_space"]["coarse_dim"])
        edge_hidden = int(self.config["edge_model"]["hidden_size"])
        coarse_bytes = coarse_len * coarse_dim * 4
        hidden_bytes = seq_len * edge_hidden * 4
        compression_ratio = hidden_bytes / max(coarse_bytes, 1)
        comm_mb = self.collab.comm.total_mb if self.mode in {"coarse_to_fine", "vanilla_collaborative_distillation"} else 0.0
        device_flops = self.device_model.flops_per_batch(int(self.config["data"]["batch_size"]), coarse_len)
        edge_flops = self.edge_model.flops_per_batch(int(self.config["data"]["batch_size"]), seq_len)
        comm_ratio = comm_mb / max((device_flops + edge_flops) / 1e9, 1e-9)
        hidden_mb = self.collab.comm.hidden_mb
        return {
            "mode": self.mode,
            "dataset": self.config["data"].get("dataset", "unknown"),
            "coarse_dim": coarse_dim,
            "compression_method": self.config["coarse_space"].get("compression_method", "linear"),
            "device_steps": int(self.config["diffusion"].get("device_steps", self.config["diffusion"].get("sampling_steps", 4))),
            "edge_steps": int(self.config["diffusion"].get("edge_steps", self.config["diffusion"].get("sampling_steps", 4))),
            "train_time": train_time,
            "step_time": train_time / max(steps, 1),
            "memory_MB": (self.coarse_space.params + self.device_model.params + self.edge_model.params) * 4 / (1024 * 1024),
            "comm_MB": comm_mb,
            "sampling_latency": sampling["sampling_latency"],
            "tokens_per_sec": sampling["tokens_per_sec"],
            "token_acc": val["token_acc"],
            "val_loss": val["val_loss"],
            "compression_ratio": compression_ratio,
            "refinement_gain": 0.0,
            "train_device_loss": losses["device"] / max(steps, 1),
            "train_edge_loss": losses["edge"] / max(steps, 1),
            "alignment_loss": val["alignment_loss"],
            "distillation_loss": losses["distill"] / max(steps, 1),
            "train_token_acc": train_acc,
            "device_params": self.device_model.params,
            "edge_params": self.edge_model.params,
            "coarse_space_params": self.coarse_space.params,
            "device_flops": device_flops,
            "edge_flops": edge_flops,
            "transmitted_coarse_representation_size": coarse_bytes,
            "transmitted_hidden_states_size": hidden_bytes,
            "communication_MB_per_batch": comm_mb / max(steps, 1),
            "communication_MB_per_generated_sequence": coarse_bytes / (1024 * 1024) if self.mode == "coarse_to_fine" else 0.0,
            "communication_computation_ratio": comm_ratio,
            "hidden_state_baseline_MB": hidden_mb,
            **sampling,
            **val,
        }

    def _save_checkpoint_metadata(self, result: Dict[str, float]) -> None:
        out = Path(self.config["outputs"]["dir"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.mode}_checkpoint.json"
        path.write_text(
            "{\n"
            + ",\n".join(f'  "{k}": "{v}"' for k, v in sorted(result.items()) if isinstance(v, (str, int, float)))
            + "\n}\n",
            encoding="utf-8",
        )


def run_mode(config: dict, mode: str, persist: bool = True) -> Dict[str, float]:
    return ExperimentTrainer(config, mode).train(persist=persist)


def with_overrides(config: dict, **kwargs) -> dict:
    import copy

    cfg = copy.deepcopy(config)
    for key, value in kwargs.items():
        if key == "coarse_dim":
            cfg["coarse_space"]["coarse_dim"] = int(value)
        elif key == "compression_method":
            cfg["coarse_space"]["compression_method"] = value
        elif key == "device_steps":
            cfg["diffusion"]["device_steps"] = int(value)
        elif key == "edge_steps":
            cfg["diffusion"]["edge_steps"] = int(value)
        elif key == "steps":
            cfg["training"]["steps"] = int(value)
    return cfg
