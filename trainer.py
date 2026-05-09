"""Training orchestration for device-only, edge-only, and collaborative modes."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from edge_device_training.collaboration import LogitDistillationCollaboration
from edge_device_training.data import build_datasets
from edge_device_training.diffusion_utils import (
    corrupt_tokens,
    denoising_loss,
    evaluate_denoising,
    measure_sampling,
    sample_noise_prob,
)
from edge_device_training.metrics import gpu_memory_mb, load_results, upsert_result, write_results
from edge_device_training.models import build_model


MODES = {"device_only", "edge_only", "collaborative"}


def choose_device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentTrainer:
    def __init__(self, config: dict, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}; expected one of {sorted(MODES)}")
        self.config = config
        self.mode = mode
        set_seed(int(config.get("seed", 7)))
        self.device = choose_device(config)
        self.train_ds, self.val_ds, self.tokenizer = build_datasets(config)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=False,
            drop_last=False,
        )
        seq_len = int(config["data"]["seq_len"])
        vocab_size = self.tokenizer.vocab_size
        pad_id = self.tokenizer.pad_token_id
        self.device_model = build_model(config["device_model"], vocab_size, seq_len, pad_id).to(self.device)
        self.edge_model = build_model(config["edge_model"], vocab_size, seq_len, pad_id).to(self.device)
        self.collaboration = LogitDistillationCollaboration(config)

    def train(self) -> Dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        steps = int(self.config["training"]["steps"])
        lr = float(self.config["training"]["lr"])
        weight_decay = float(self.config["training"].get("weight_decay", 0.0))
        device_opt = torch.optim.AdamW(self.device_model.parameters(), lr=lr, weight_decay=weight_decay)
        edge_opt = torch.optim.AdamW(self.edge_model.parameters(), lr=lr, weight_decay=weight_decay)
        train_iter = self._infinite_train_batches()

        self.device_model.train()
        self.edge_model.train()
        total_loss = 0.0
        total_device_loss = 0.0
        total_edge_loss = 0.0
        start = time.perf_counter()

        for step in range(1, steps + 1):
            clean_ids = next(train_iter).to(self.device)
            noise_prob = sample_noise_prob(clean_ids.size(0), self.config, self.device)
            corrupted_ids, target_mask = corrupt_tokens(
                clean_ids,
                self.tokenizer.mask_token_id,
                self.tokenizer.pad_token_id,
                noise_prob,
            )

            if self.mode == "device_only":
                device_opt.zero_grad(set_to_none=True)
                logits = self.device_model(corrupted_ids, noise_prob)
                loss, _ = denoising_loss(logits, clean_ids, target_mask)
                loss.backward()
                device_opt.step()
                total_device_loss += float(loss.item())
                total_loss += float(loss.item())

            elif self.mode == "edge_only":
                edge_opt.zero_grad(set_to_none=True)
                logits = self.edge_model(corrupted_ids, noise_prob)
                loss, _ = denoising_loss(logits, clean_ids, target_mask)
                loss.backward()
                edge_opt.step()
                total_edge_loss += float(loss.item())
                total_loss += float(loss.item())

            else:
                edge_opt.zero_grad(set_to_none=True)
                edge_logits = self.edge_model(corrupted_ids, noise_prob)
                edge_loss, _ = denoising_loss(edge_logits, clean_ids, target_mask)
                edge_loss.backward()
                edge_opt.step()

                device_opt.zero_grad(set_to_none=True)
                device_logits = self.device_model(corrupted_ids, noise_prob)
                device_loss, _ = denoising_loss(device_logits, clean_ids, target_mask)
                with torch.no_grad():
                    teacher_logits = self.edge_model(corrupted_ids, noise_prob)
                kd_loss, _ = self.collaboration.distill(
                    step,
                    device_logits,
                    teacher_logits,
                    clean_ids,
                    corrupted_ids,
                    target_mask,
                )
                loss = device_loss + kd_loss
                loss.backward()
                device_opt.step()

                total_device_loss += float(device_loss.item())
                total_edge_loss += float(edge_loss.item())
                total_loss += float(loss.item())

        train_time = time.perf_counter() - start
        eval_model = self.edge_model if self.mode == "edge_only" else self.device_model
        val = evaluate_denoising(
            eval_model,
            self.val_loader,
            self.config,
            self.tokenizer,
            self.device,
            max_batches=int(self.config["training"].get("eval_batches", 4)),
        )
        sampling = measure_sampling(eval_model, self.config, self.tokenizer, self.device)
        comm_mb = self.collaboration.comm.total_mb if self.mode == "collaborative" else 0.0
        result = {
            "mode": self.mode,
            "train_loss": total_loss / max(steps, 1),
            "device_train_loss": total_device_loss / max(steps, 1),
            "edge_train_loss": total_edge_loss / max(steps, 1),
            "train_time": train_time,
            "step_time": train_time / max(steps, 1),
            "memory_MB": gpu_memory_mb(self.device),
            "comm_MB": comm_mb,
            "uploaded_MB": self.collaboration.comm.uploaded_bytes / (1024 * 1024),
            "downloaded_MB": self.collaboration.comm.downloaded_bytes / (1024 * 1024),
            "sync_rounds": float(self.collaboration.comm.sync_rounds if self.mode == "collaborative" else 0),
            **sampling,
            **val,
        }
        self._save_checkpoint(eval_model)
        output_dir = self.config["outputs"]["dir"]
        results = upsert_result(load_results(output_dir), result)
        write_results(results, output_dir)
        return result

    def _infinite_train_batches(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def _save_checkpoint(self, model) -> None:
        output_dir = Path(self.config["outputs"]["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "model_state": model.state_dict(),
                "vocab": self.tokenizer.token_to_id,
            },
            output_dir / f"{self.mode}_checkpoint.pt",
        )

