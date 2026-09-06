#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [str(ROOT / "src")] + [
    p for p in sys.path if Path(p).resolve() != SCRIPT_DIR and p not in ("", str(ROOT / "src"))
]

from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main() -> int:
    path = Path(__file__).resolve().parent / "the-verdict.txt"
    text = path.read_text(encoding="utf-8")

    tokenizer = BPETokenizer("gpt2")
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    sys.stdout.write(f"Decoded text matches original: {decoded == text}\n")
    sys.stdout.write(f"Token count: {len(token_ids)}\n")
    print(tokenizer.decode(tokenizer.encode("John Doe is a software engineer.")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
