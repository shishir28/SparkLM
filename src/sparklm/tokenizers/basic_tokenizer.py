import re
from typing import Dict, List, Pattern


class BasicTokenizer:
    """A minimal tokenizer that splits text into chunks and can encode/decode IDs."""

    def __init__(self, pattern: str):
        if not pattern:
            raise ValueError("A regex pattern must be provided to initialize the tokenizer.")
        self._pattern_str: str = pattern
        self._pattern: Pattern[str] = re.compile(pattern)
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}

    def set_pattern(self, pattern: str) -> None:
        self._pattern_str = pattern
        self._pattern = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        """Split text on the configured regex separator pattern and drop empty parts."""
        return [part for part in self._pattern.split(text) if part != ""]

    def _build_vocab(self, tokens: List[str]) -> None:
        vocab = sorted(set(tokens))
        self.str_to_int = {token: index for index, token in enumerate(vocab)}
        self.int_to_str = {index: token for token, index in self.str_to_int.items()}

    def encode(self, text: str) -> List[int]:
        """Tokenize text and convert each token to an integer ID."""
        tokens = self.tokenize(text)
        if not tokens:
            return []

        self._build_vocab(tokens)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text using the same vocabulary."""
        if not ids:
            return ""
        tokens = [self.int_to_str[index] for index in ids]
        return " ".join(tokens)

    @property
    def pattern(self) -> str:
        return self._pattern_str
