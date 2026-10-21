"""Compute initial train and validation losses on The Verdict.

Run from the repository root with:
    .venv/bin/python scripts/loss_smoke.py
"""

from pathlib import Path

import torch

from sparklm.config import GPT_CONFIG_124M
from sparklm.data_loader import create_dataloader_v1
from sparklm.loss import calc_loss_loader
from sparklm.model import GPTModel
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main() -> int:
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer("gpt2")
    text = (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")
    train_ratio = 0.90
    split_index = int(len(text) * train_ratio)
    train_text = text[:split_index]
    val_text = text[split_index:]
    context_length = GPT_CONFIG_124M["context_length"]

    total_tokens = len(tokenizer.encode(text))
    assert total_tokens * train_ratio >= context_length

    loader_kwargs = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "max_length": context_length,
        "stride": context_length,
        "num_workers": 0,
    }
    train_loader = create_dataloader_v1(
        train_text, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = create_dataloader_v1(
        val_text, shuffle=False, drop_last=False, **loader_kwargs
    )

    config = dict(GPT_CONFIG_124M)
    config.update({"embedding_dim": 32, "num_layers": 2, "num_heads": 4})
    model = GPTModel(config).to(device).eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"device: {device}")
    print(f"train windows: {len(train_loader.dataset)}")
    print(f"validation windows: {len(val_loader.dataset)}")
    print(f"initial train loss: {train_loss:.4f}")
    print(f"initial validation loss: {val_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
