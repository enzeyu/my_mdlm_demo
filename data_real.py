"""Dataset and tokenizer utilities for DART LoRA training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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


def load_tokenizer(tokenizer_name: str, local_files_only: bool = False):
    """Load GPT-style tokenizer and add a real `[MASK]` token when absent."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    return tokenizer


def _wikitext_config_name(dataset_name: str) -> str | None:
    if dataset_name in {"wikitext-2", "wikitext2"}:
        return "wikitext-2-raw-v1"
    if dataset_name in {"wikitext-103", "wikitext103"}:
        return "wikitext-103-raw-v1"
    return None


def _local_wikitext_parquet_files(config_name: str, cache_dir: str | None) -> dict[str, list[str]] | None:
    """Find manually downloaded WikiText parquet files from hf_downloads/datasets."""
    if not cache_dir:
        return None
    cache_path = Path(cache_dir)
    candidates = [
        cache_path / config_name,
        cache_path / "wikitext" / config_name,
    ]
    local_dir = next((path for path in candidates if path.exists()), None)
    if local_dir is None:
        return None

    data_files = {}
    for split in ["train", "validation", "test"]:
        files = sorted(str(path) for path in local_dir.rglob(f"{split}-*.parquet"))
        if files:
            data_files[split] = files
    if "train" not in data_files or ("validation" not in data_files and "test" not in data_files):
        raise RuntimeError(
            f"Local WikiText directory {local_dir} is incomplete; expected train and validation/test parquet files."
        )
    return data_files


def _load_local_wikitext_parquet(config_name: str, cache_dir: str | None):
    """Load manually downloaded WikiText parquet files from hf_downloads/datasets."""
    data_files = _local_wikitext_parquet_files(config_name, cache_dir)
    if data_files is None:
        return None

    from datasets import load_dataset

    cache_path = Path(cache_dir) if cache_dir else Path("/tmp")
    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_path / "parquet_cache" / config_name),
        verification_mode="no_checks",
    )


def _read_parquet_texts(files: list[str], limit: int) -> list[str]:
    """Read text rows from parquet directly, avoiding datasets/pandas cache paths."""
    import pyarrow.parquet as pq

    texts = []
    for file_path in files:
        table = pq.ParquetFile(file_path).read(columns=["text"])
        for value in table.column("text").to_pylist():
            text = str(value).strip() if value is not None else ""
            if text:
                texts.append(text)
            if len(texts) >= limit:
                return texts
    return texts


def load_wikitext_splits(dataset_name: str, max_train_examples: int, max_val_examples: int, cache_dir: str | None = None):
    """Load WikiText-style train/validation text splits from HuggingFace datasets."""
    load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    config_name = _wikitext_config_name(dataset_name)
    if config_name:
        local_files = _local_wikitext_parquet_files(config_name, cache_dir)
        if local_files is not None:
            val_key = "validation" if "validation" in local_files else "test"
            train_texts = _read_parquet_texts(local_files["train"], max_train_examples)
            val_texts = _read_parquet_texts(local_files[val_key], max_val_examples)
            return train_texts, val_texts
        from datasets import load_dataset

        dataset = load_dataset("wikitext", config_name, **load_kwargs)
    else:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, **load_kwargs)

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
    dataset_cache_dir = config.get("dataset_cache_dir", "/mnt/data/enzeyu/hf_downloads/datasets")
    os.environ.setdefault("HF_DATASETS_CACHE", str(dataset_cache_dir))
    tokenizer = load_tokenizer(
        config["tokenizer_name"],
        local_files_only=bool(config.get("hf_local_files_only", False)),
    )
    train_texts, val_texts = load_wikitext_splits(
        config.get("dataset_name", "wikitext-2"),
        int(config.get("max_train_examples", 2000)),
        int(config.get("max_val_examples", 500)),
        cache_dir=str(dataset_cache_dir),
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
