"""Data and tokenizer utilities for the minimal training prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


TOY_CORPUS = [
    "edge devices train small diffusion language models on private text",
    "edge servers guide device models with soft token targets",
    "masked diffusion language models recover corrupted tokens",
    "collaborative training trades communication for better token recovery",
    "device models upload uncertain positions to the edge server",
    "edge models return logits for high entropy masked tokens",
    "periodic synchronization improves small model denoising quality",
    "communication aware training measures uploaded and downloaded bytes",
    "local text remains on the device while compact signals are shared",
    "research prototypes compare device only edge only and collaborative modes",
    "diffusion sampling fills masked tokens over multiple denoising steps",
    "small models are fast but large models provide stronger feedback",
]


@dataclass
class SimpleTokenizer:
    """A tiny word-level tokenizer with the subset of the HF tokenizer API we need."""

    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    unk_token: str = "[UNK]"

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> "SimpleTokenizer":
        special = ["[PAD]", "[MASK]", "[UNK]"]
        vocab = list(special)
        seen = set(vocab)
        for text in texts:
            for token in text.lower().split():
                if token not in seen:
                    seen.add(token)
                    vocab.append(token)
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.mask_token]

    def encode(self, text: str) -> List[int]:
        unk = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk) for token in text.lower().split()]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.id_to_token.get(int(idx), self.unk_token) for idx in ids)


class TokenChunkDataset(Dataset):
    """Fixed-length token chunks for denoising training."""

    def __init__(self, chunks: torch.Tensor):
        self.chunks = chunks.long()

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.chunks[index]


def load_texts(data_config: dict) -> List[str]:
    """Load built-in toy text now; keep hooks for local/wikitext-style data later."""
    dataset = data_config.get("dataset", "toy")
    if dataset == "toy":
        repeat = int(data_config.get("toy_repeat", 8))
        return TOY_CORPUS * repeat

    if dataset == "local_file":
        path = Path(data_config["path"])
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if dataset == "wikitext2":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` to use wikitext2, or set dataset=toy/local_file.") from exc
        split = data_config.get("split", "train")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return [row["text"].strip() for row in ds if row["text"].strip()]

    raise ValueError(f"Unknown dataset: {dataset}")


def build_tokenizer(data_config: dict, texts: Sequence[str]) -> SimpleTokenizer:
    """Build the default simple tokenizer; HF tokenizers can be added behind this hook."""
    tokenizer_name = data_config.get("tokenizer", "simple")
    if tokenizer_name != "simple":
        raise ValueError("The toy prototype currently supports tokenizer: simple.")
    return SimpleTokenizer.from_texts(texts)


def make_chunks(token_ids: List[int], seq_len: int, pad_id: int) -> torch.Tensor:
    if not token_ids:
        raise ValueError("No tokens available for dataset construction.")
    chunks = []
    stride = seq_len
    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start : start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
        chunks.append(chunk)
    return torch.tensor(chunks, dtype=torch.long)


def build_datasets(config: dict) -> Tuple[TokenChunkDataset, TokenChunkDataset, SimpleTokenizer]:
    data_config = config["data"]
    texts = load_texts(data_config)
    tokenizer = build_tokenizer(data_config, texts)
    token_ids: List[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
    chunks = make_chunks(token_ids, int(data_config["seq_len"]), tokenizer.pad_token_id)
    if chunks.size(0) < 2:
        chunks = chunks.repeat(2, 1)

    generator = torch.Generator().manual_seed(int(config.get("seed", 7)))
    perm = torch.randperm(chunks.size(0), generator=generator)
    chunks = chunks[perm]
    n_val = max(1, int(chunks.size(0) * float(data_config.get("val_fraction", 0.2))))
    val_chunks = chunks[:n_val]
    train_chunks = chunks[n_val:]
    if train_chunks.size(0) == 0:
        train_chunks = val_chunks.clone()
    return TokenChunkDataset(train_chunks), TokenChunkDataset(val_chunks), tokenizer

