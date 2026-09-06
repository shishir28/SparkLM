#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [str(ROOT / "src")] + [
    p for p in sys.path if Path(p).resolve() != SCRIPT_DIR and p not in ("", str(ROOT / "src"))
]

from sparklm.data_loader import create_dataloader_v1
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main() -> int:
    path = Path(__file__).resolve().parent / "the-verdict.txt"
    text = path.read_text(encoding="utf-8")

    loader = create_dataloader_v1(
        text=text,
        tokenizer=BPETokenizer("gpt2"),
        batch_size=1,
        max_length=4, 
        stride=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(loader))
    print(f"Batch x: {batch_x}")
    print(f"Batch y: {batch_y}")
    # print(f"x shape: {tuple(batch_x.shape)}")
    # print(f"y shape: {tuple(batch_y.shape)}")
    # print(f"x sample: {batch_x[0].tolist()[:10]}")
    # print(f"y sample: {batch_y[0].tolist()[:10]}")
    # Increase the stride to avoid overlapping sequences as more overlapping sequences can lead to increased overfitting
    another_loader = create_dataloader_v1(
        text=text,
        tokenizer=BPETokenizer("gpt2"),
        batch_size=4, # number of rows in _batch_x and _batch_y
        max_length=8, # number of columns in _batch_x and _batch_y
        stride=2, # how far the window moves
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batch_x2, batch_y2 = next(iter(another_loader))
    print(f"Batch x2: {batch_x2}")
    print(f"Batch y2: {batch_y2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
