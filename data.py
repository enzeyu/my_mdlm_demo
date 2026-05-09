"""Dataset and tokenizer utilities for the coarse-to-fine MDLM demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


TOY_CORPUS = [
    "edge devices train small diffusion language models on private text",
    "edge servers guide device models with compact semantic signals",
    "masked diffusion language models recover corrupted tokens in parallel",
    "coarse representations reduce hidden state communication cost",
    "edge refinement restores fine grained lexical details",
    "collaborative training compares quality latency and communication",
    "token masking creates a denoising objective for language generation",
    "small models produce low dimensional semantic sketches",
    "large models condition on sketches and refine token predictions",
    "research prototypes measure pareto trade offs across systems metrics",
]

FALLBACK_REAL_TEXT = [
    "Natural language processing studies algorithms that represent and generate text.",
    "Wikipedia articles contain factual paragraphs with varied vocabulary and syntax.",
    "A diffusion language model learns to reconstruct masked tokens over repeated denoising steps.",
    "Efficient edge intelligence often compresses intermediate representations before transmission.",
    "Transformer networks use attention layers to mix information across sequence positions.",
    "Model evaluation should report accuracy, validation loss, latency, memory, and communication cost.",
    "Coarse semantic spaces can preserve topic information while discarding lexical detail.",
    "The edge server can refine compact device outputs into fluent token sequences.",
]


@dataclass
class SimpleTokenizer:
    """Small word tokenizer with the subset of the HuggingFace API used here."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    mask_token: str = "[MASK]"

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> "SimpleTokenizer":
        vocab = ["[PAD]", "[UNK]", "[MASK]"]
        seen = set(vocab)
        for text in texts:
            for tok in text.lower().replace("\n", " ").split():
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        return cls(token_to_id, id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.mask_token]

    def encode(self, text: str) -> List[int]:
        unk = self.unk_token_id
        return [self.token_to_id.get(tok, unk) for tok in text.lower().replace("\n", " ").split()]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.id_to_token.get(int(idx), self.unk_token) for idx in ids)


class HFTokenizerAdapter:
    """Adapter that gives GPT-2 style tokenizers a usable mask token id."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._base_vocab = int(tokenizer.vocab_size)
        self._pad = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        self._mask = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else self._base_vocab

    @property
    def vocab_size(self) -> int:
        return self._base_vocab

    @property
    def total_vocab_size(self) -> int:
        return self._base_vocab + (1 if self._mask == self._base_vocab else 0)

    @property
    def pad_token_id(self) -> int:
        return int(self._pad)

    @property
    def mask_token_id(self) -> int:
        return int(self._mask)

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Iterable[int]) -> str:
        clean = [int(idx) for idx in ids if int(idx) < self._base_vocab]
        return self.tokenizer.decode(clean)


def _try_hf_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer

        return HFTokenizerAdapter(AutoTokenizer.from_pretrained(name, local_files_only=True))
    except Exception:
        return None


def load_texts(data_config: dict) -> List[str]:
    dataset = data_config.get("dataset", "toy")
    max_chars = data_config.get("max_chars")
    max_examples = int(data_config.get("max_examples", 256))

    if dataset == "toy":
        return TOY_CORPUS * int(data_config.get("toy_repeat", 8))

    if dataset == "local_file":
        path = Path(data_config["path"])
        text = path.read_text(encoding="utf-8")
        return _split_text(text[:max_chars] if max_chars else text, max_examples)

    if dataset in {"arche_like", "tinystories"}:
        local_mix = Path(data_config.get("path", "data/training_mix.txt"))
        if local_mix.exists():
            text = local_mix.read_text(encoding="utf-8")
            return _split_text(text[:max_chars] if max_chars else text, max_examples)
        texts = _load_hf_text("roneneldan/TinyStories", None, "train", "text", max_examples)
        if texts:
            return texts
        texts = _load_hf_text("wikitext", "wikitext-2-raw-v1", "train", "text", max_examples)
        return texts or FALLBACK_REAL_TEXT * 16

    if dataset == "wikitext2":
        texts = _load_hf_text("wikitext", "wikitext-2-raw-v1", data_config.get("split", "train"), "text", max_examples)
        return texts or FALLBACK_REAL_TEXT * 16

    raise ValueError(f"Unknown dataset: {dataset}")


def _load_hf_text(name: str, subset: str | None, split: str, field: str, max_examples: int) -> List[str]:
    try:
        from datasets import load_dataset

        kwargs = {"split": split}
        ds = load_dataset(name, subset, **kwargs) if subset else load_dataset(name, **kwargs)
        texts = []
        for row in ds:
            text = str(row.get(field, "")).strip()
            if text:
                texts.append(text)
            if len(texts) >= max_examples:
                break
        return texts
    except Exception:
        return []


def _split_text(text: str, max_examples: int) -> List[str]:
    chunks = [part.strip() for part in text.split("\n") if part.strip()]
    if not chunks:
        chunks = [text.strip()] if text.strip() else []
    return chunks[:max_examples]


def build_tokenizer(data_config: dict, texts: Sequence[str]):
    name = data_config.get("tokenizer", "simple")
    if name != "simple":
        tok = _try_hf_tokenizer(name)
        if tok is not None:
            return tok
    return SimpleTokenizer.from_texts(texts)


def make_chunks(token_ids: List[int], seq_len: int, pad_id: int, max_chunks: int | None = None) -> np.ndarray:
    if not token_ids:
        raise ValueError("No tokens available for dataset construction.")
    chunks = []
    for start in range(0, len(token_ids), seq_len):
        chunk = token_ids[start : start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
        chunks.append(chunk)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return np.asarray(chunks, dtype=np.int64)


def build_datasets(config: dict) -> Tuple[np.ndarray, np.ndarray, object]:
    data_config = config["data"]
    texts = load_texts(data_config)
    tokenizer = build_tokenizer(data_config, texts)
    token_ids: List[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
    if len(token_ids) < int(data_config["seq_len"]):
        token_ids = token_ids * (int(data_config["seq_len"]) // max(len(token_ids), 1) + 2)

    chunks = make_chunks(
        token_ids,
        int(data_config["seq_len"]),
        int(tokenizer.pad_token_id),
        data_config.get("max_chunks"),
    )
    if len(chunks) < 2:
        chunks = np.repeat(chunks, 2, axis=0)

    rng = np.random.default_rng(int(config.get("seed", 7)))
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * float(data_config.get("val_fraction", 0.2))))
    val = chunks[:n_val]
    train = chunks[n_val:]
    if len(train) == 0:
        train = val.copy()
    return train, val, tokenizer
