from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


class GPTDatasetV1(Dataset):
    """Create fixed-length input/target windows for next-token prediction."""

    def __init__(self, text: str, tokenizer: BPETokenizer, max_length: int = 512, stride: int = 128):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.input_ids = []
        self.target_ids = []

        # Encode the raw text once; every window is built from this token stream.
        self.token_ids = tokenizer.encode(text)

        # Slide a window across the token stream; target is the same window shifted by one token.
        for i in range(0, len(self.token_ids) - max_length + 1, stride):
            input_chunk = self.token_ids[i : i + max_length]
            target_chunk = self.token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


