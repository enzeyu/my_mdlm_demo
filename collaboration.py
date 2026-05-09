"""Coarse-to-fine collaboration utilities and communication accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommunicationStats:
    coarse_bytes: int = 0
    hidden_bytes: int = 0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    sync_rounds: int = 0

    @property
    def total_bytes(self) -> int:
        return self.coarse_bytes + self.uploaded_bytes + self.downloaded_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def hidden_mb(self) -> float:
        return self.hidden_bytes / (1024 * 1024)


class CoarseToFineCollaboration:
    """Tracks coarse representation transfer instead of dense hidden transfer."""

    def __init__(self):
        self.comm = CommunicationStats()

    def record_coarse(self, batch: int, coarse_len: int, coarse_dim: int):
        self.comm.coarse_bytes += int(batch * coarse_len * coarse_dim * 4)
        self.comm.sync_rounds += 1

    def record_hidden_baseline(self, batch: int, seq_len: int, hidden_size: int):
        self.comm.hidden_bytes += int(batch * seq_len * hidden_size * 4)

    def record_distillation(self, selected_tokens: int, vocab_size: int):
        self.comm.uploaded_bytes += int(selected_tokens * 3 * 8)
        self.comm.downloaded_bytes += int(selected_tokens * vocab_size * 4)
        self.comm.sync_rounds += 1
