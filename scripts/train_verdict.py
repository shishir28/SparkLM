"""Train SparkLM on The Verdict.

Run from the repository root with:
    .venv/bin/python scripts/train_verdict.py
"""


from pathlib import Path

import torch
import time

from sparklm.config import GPT_CONFIG_124M
from sparklm.data_loader import create_dataloader_v1
from sparklm.model import GPTModel
from sparklm.plotting import plot_losses
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer
from sparklm.training import train_model_simple


def main() -> int:
    # Set the random seed and select CPU or CUDA.
    start_time = time.time()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and scripts/the-verdict.txt.
    path = Path(__file__).resolve().parent / "the-verdict.txt"
    text_data = path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer("gpt2")

    # Split the text into 90% training data and 10% validation data.
    train_ratio = 0.90
    split_index = int(len(text_data) * train_ratio)
    train_text = text_data[:split_index]
    val_text = text_data[split_index:]

    # Build the training and validation data loaders.
    train_loader = create_dataloader_v1(
        text=train_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"]    ,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    # Check that the training split has at least one context-length window.
    train_token_count = len(tokenizer.encode(train_text))
    assert train_token_count >= GPT_CONFIG_124M["context_length"], "Training split is too small for the context length."

    # Build the GPT model and move it to the selected device.
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # Create AdamW after moving the model to the device.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds on device {device}.")


    plot_losses(
        epochs_seen=list(range(len(train_losses))),
        tokens_seen=track_tokens_seen,
        train_losses=train_losses,
        val_losses=val_losses,
        path=  Path(__file__).resolve().parents[1] / "experiments" / "runs" / "loss_plot.png"
    )
    # Plot losses under experiments/runs/.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
