#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]
sys.path.insert(0, str(ROOT / "src"))

import torch

from sparklm.data_loader import create_dataloader_v1
from sparklm.embeddings import PositionalEmbedding, TokenEmbedding
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
    batch_x, _ = next(iter(loader))

    token_embedding = TokenEmbedding(vocab_size=50257, embedding_dim=256)
    token_embeddings = token_embedding(batch_x)
    print(f"Token embeddings shape: {tuple(token_embeddings.shape)}")

    positions = torch.arange(batch_x.shape[1])
    positional_embedding = PositionalEmbedding(max_seq_len=4, embedding_dim=256)
    positional_embeddings = positional_embedding(positions)
    print(f"Positional embeddings shape: {tuple(positional_embeddings.shape)}")

    combined = token_embeddings + positional_embeddings.unsqueeze(0)
    print(f"Combined embeddings shape: {tuple(combined.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
