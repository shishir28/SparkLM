import json
from collections import Counter
from typing import Iterable, List, Optional, Dict


class CharTokenizer:
    """Deterministic character tokenizer.

    Specials: <unk> and <eot> are included by default with stable IDs.
    Vocab is specials followed by sorted characters (by Unicode codepoint).
    """

    UNK = "<unk>"
    EOT = "<eot>"

    def __init__(self, chars: List[str], specials: Optional[List[str]] = None):
        if specials is None:
            specials = [self.UNK, self.EOT]
        # Ensure specials are first and stable
        self.specials = list(specials)
        # chars should not contain specials
        chars = [c for c in chars if c not in self.specials]
        self.chars = sorted(chars)
        self.vocab: List[str] = self.specials + self.chars
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: s for i, s in enumerate(self.vocab)}
        self.unk_id = self.stoi[self.UNK]
        self.eot_id = self.stoi[self.EOT]

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], min_freq: int = 1):
        counter = Counter()
        for t in texts:
            counter.update(t)
        chars = [c for c, freq in counter.items() if freq >= min_freq]
        # deterministic sort is applied in __init__
        return cls(chars)

    def encode(self, text: str, add_eot: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eot:
            ids.append(self.eot_id)
        return ids

    def decode(self, ids: Iterable[int], strip_eot: bool = True) -> str:
        chars = []
        for i in ids:
            # If unknown id -> UNK token text
            ch = self.itos.get(i, self.UNK)
            if strip_eot and i == self.eot_id:
                break
            chars.append(ch)
        return "".join(chars)

    def save(self, path: str):
        payload = {
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = payload["vocab"]
        # specials assumed first two
        specials = vocab[:2]
        chars = vocab[2:]
        return cls(chars, specials=specials)
