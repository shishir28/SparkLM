from pathlib import Path

import torch

from sparklm.checkpoint import load_checkpoint, save_checkpoint
from sparklm.data_loader import create_dataloader_v1
from sparklm.model import GPTModel
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer
from sparklm.training import train_model_simple


def states_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            states_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def main() -> int:
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(__file__).resolve().parent / "the-verdict.txt"
    text_data = path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer("gpt2")
    config = {
        "vocab_size": 50257,
        "context_length": 4,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }

    train_ratio = 0.90
    split_index = int(len(text_data) * train_ratio)
    train_text = text_data[:split_index]
    val_text = text_data[split_index:]

    loader_kwargs = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "max_length": config["context_length"],
        "stride": config["context_length"],
        "num_workers": 0,
    }
    train_loader = create_dataloader_v1(
        train_text,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = create_dataloader_v1(
        val_text,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    train_token_count = len(tokenizer.encode(train_text))
    assert train_token_count >= config["context_length"]

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )
    _, _, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    save_path = Path(__file__).resolve().parent / "checkpoint_smoke_test.pt"
    metadata = {"epoch": 1, "tokens_seen": track_tokens_seen[-1]}
    save_checkpoint(save_path, model, optimizer, config, metadata)

    loaded_model = GPTModel(config).to(device)
    loaded_optimizer = torch.optim.AdamW(
        loaded_model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )
    loaded_metadata = load_checkpoint(
        save_path,
        loaded_model,
        optimizer=loaded_optimizer,
        map_location=device,
    )

    print("Comparing logits and loss between original and loaded model...")
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            original_logits = model(input_batch)
            loaded_logits = loaded_model(input_batch)
            if not torch.allclose(original_logits, loaded_logits, atol=1e-6):
                print("Logits do not match!")
                return 1

            original_loss = torch.nn.functional.cross_entropy(
                original_logits.view(-1, original_logits.size(-1)),
                target_batch.view(-1),
            )
            loaded_loss = torch.nn.functional.cross_entropy(
                loaded_logits.view(-1, loaded_logits.size(-1)),
                target_batch.view(-1),
            )
            if not torch.isclose(original_loss, loaded_loss, atol=1e-6):
                print("Loss does not match!")
                return 1

    if not states_equal(loaded_optimizer.state_dict(), optimizer.state_dict()):
        print("Optimizer state does not match!")
        return 1
    if loaded_metadata != metadata:
        print("Metadata does not match!")
        return 1

    print("Checkpoint smoke test passed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
