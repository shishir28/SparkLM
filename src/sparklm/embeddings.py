from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Learn a vector for each token ID in a vocabulary."""

    def __init__(self, vocab_size: int, embedding_dim: int | None = None, *, output_dim: int | None = None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = output_dim
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided")

        self.vocab_size = vocab_size  # number of rows in the embedding matrix
        self.embedding_dim = embedding_dim  # number of columns in the embedding matrix
        self.output_dim = embedding_dim  # compatibility alias
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Learn a vector for each absolute position in a sequence."""

    def __init__(
        self,
        max_seq_len: int | None = None,
        embedding_dim: int | None = None,
        *,
        context_length: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()

        if max_seq_len is None:
            max_seq_len = context_length
        if embedding_dim is None:
            embedding_dim = output_dim
        if max_seq_len is None:
            raise ValueError("max_seq_len or context_length must be provided")
        if embedding_dim is None:
            raise ValueError("embedding_dim or output_dim must be provided")

        self.max_seq_len = max_seq_len  # number of rows in the embedding matrix
        self.embedding_dim = embedding_dim  # number of columns in the embedding matrix
        self.output_dim = embedding_dim  # compatibility alias
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)
