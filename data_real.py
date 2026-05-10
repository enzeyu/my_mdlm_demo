"""Real dataset and tokenizer utilities for coarse-to-fine diffusion LM training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TokenizerInfo:
    """Small bundle of tokenizer ids used by the denoising code."""

    vocab_size: int
    pad_token_id: int
    mask_token_id: int


class TokenBlockDataset(Dataset):
    """Fixed-length token blocks for masked diffusion denoising."""

    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks.long()

    def __len__(self) -> int:
        """Return the number of fixed-length token blocks."""
        return self.blocks.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one token block."""
        return self.blocks[index]


def load_tokenizer(tokenizer_name: str):
    """Load GPT-style tokenizer and add a real `[MASK]` token when absent."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    return tokenizer


def load_wikitext_splits(dataset_name: str, max_train_examples: int, max_val_examples: int):
    """Load WikiText-style train/validation text splits from HuggingFace datasets."""
    from datasets import load_dataset

    if dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    else:
        dataset = load_dataset(dataset_name)

    train_texts = _take_nonempty(dataset["train"], max_train_examples)
    val_split = "validation" if "validation" in dataset else "test"
    val_texts = _take_nonempty(dataset[val_split], max_val_examples)
    return train_texts, val_texts


def _take_nonempty(split, limit: int):
    """Collect non-empty text examples from a HuggingFace split."""
    texts = []
    for row in split:
        text = row.get("text", "").strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def tokenize_to_blocks(texts, tokenizer, max_length: int, max_blocks: int) -> torch.Tensor:
    """Concatenate texts, tokenize with GPT-2, and cut into fixed-length blocks."""
    token_ids = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
        token_ids.append(tokenizer.eos_token_id)

    if len(token_ids) < max_length:
        raise RuntimeError("Dataset produced too few tokens. Increase max examples or check dataset loading.")

    usable = (len(token_ids) // max_length) * max_length
    token_ids = token_ids[:usable]
    blocks = torch.tensor(token_ids, dtype=torch.long).view(-1, max_length)
    if max_blocks > 0:
        blocks = blocks[:max_blocks]
    return blocks


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, object, TokenizerInfo]:
    """Build tokenizer plus train/validation dataloaders from the YAML config."""
    tokenizer = load_tokenizer(config["tokenizer_name"])
    train_texts, val_texts = load_wikitext_splits(
        config.get("dataset_name", "wikitext-2"),
        int(config.get("max_train_examples", 2000)),
        int(config.get("max_val_examples", 500)),
    )
    max_length = int(config["max_length"])
    train_blocks = tokenize_to_blocks(train_texts, tokenizer, max_length, int(config.get("max_train_blocks", 4096)))
    val_blocks = tokenize_to_blocks(val_texts, tokenizer, max_length, int(config.get("max_val_blocks", 512)))

    train_loader = DataLoader(
        TokenBlockDataset(train_blocks),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        TokenBlockDataset(val_blocks),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=True,
    )
    info = TokenizerInfo(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        mask_token_id=int(tokenizer.mask_token_id),
    )
    return train_loader, val_loader, tokenizer, info


def mask_tokens(input_ids: torch.Tensor, mask_token_id: int, pad_token_id: int, mask_ratio: float):
    """Apply masked diffusion corruption and return corrupted ids plus target mask."""
    valid = input_ids.ne(pad_token_id)
    target_mask = (torch.rand(input_ids.shape, device=input_ids.device) < mask_ratio) & valid
    noised = input_ids.clone()
    noised[target_mask] = mask_token_id

    # Ensure every sequence has at least one supervised denoising position.
    missing = ~target_mask.any(dim=1)
    if missing.any():
        rows = torch.where(missing)[0]
        for row in rows.tolist():
            valid_pos = torch.where(valid[row])[0]
            if valid_pos.numel() > 0:
                pos = valid_pos[torch.randint(valid_pos.numel(), (1,), device=input_ids.device)]
                target_mask[row, pos] = True
                noised[row, pos] = mask_token_id
    return noised, target_mask
