class CharacterTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {}
        self.itos = {}
        for i, ch in enumerate(chars):
            self.stoi[ch] = i
            self.itos[i] = ch

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character: {ch!r}")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i not in self.itos:
                raise ValueError(f"Unknown id: {i!r}")
            chars.append(self.itos[i])
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
