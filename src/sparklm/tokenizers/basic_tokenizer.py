import re
from typing import List, Optional, Pattern


class BasicTokenizer:
    """A minimal, configurable tokenizer that uses a regex pattern to extract
    tokens from an input string. The user can provide a regex pattern or set it
    later via set_pattern().

    This file intentionally keeps the implementation small: you will provide the
    final regex for the tokenization behavior.
    """

    def __init__(self, pattern: Optional[str] = None):
        self._pattern_str: Optional[str] = pattern
        self._pattern: Optional[Pattern[str]] = re.compile(pattern) if pattern else None

    def set_pattern(self, pattern: str) -> None:
        """Set or replace the regex pattern used for tokenization."""
        self._pattern_str = pattern
        self._pattern = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        """Split text on the configured regex separator pattern and drop empty parts."""
        if self._pattern is None:
            raise ValueError("No regex pattern set for tokenizer. Call set_pattern() or pass a pattern to __init__.")
        return [part for part in self._pattern.split(text) if part != ""]

    @property
    def pattern(self) -> Optional[str]:
        return self._pattern_str
