from __future__ import annotations

from typing import List

import tiktoken


class BPETokenizer:
    """Thin wrapper around GPT-2 BPE tokenization via tiktoken."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._encoder = tiktoken.get_encoding(model_name)

    @property
    def encoder(self):
        return self._encoder

    def encode(self, text: str) -> List[int]:
        return self._encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) -> str:
        return self._encoder.decode(ids)

    def vocab_size(self) -> int:
        return self._encoder.n_vocab


# Compatibility alias for older imports.
TiktokenTokenizer = BPETokenizer
