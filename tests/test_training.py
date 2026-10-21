import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sparklm.training import (
    evaluate_model,
    generate_and_print_sample,
    train_model_simple,
)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=4, context_length=4):
        super().__init__()
        self.pos_embedding = nn.Embedding(context_length, 1)
        self.logits = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape
        return self.logits.expand(batch_size, sequence_length, -1)


class DummyTokenizer:
    def encode(self, text, allowed_special=None):
        return [0]

    def decode(self, ids):
        return "generated " + " ".join(str(token_id) for token_id in ids)


def make_loader(target_value=0):
    inputs = torch.tensor([[0, 1]])
    targets = torch.full_like(inputs, target_value)
    return DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)


def test_evaluate_model_restores_train_mode_without_gradients():
    model = TinyLanguageModel()
    model.train()

    train_loss, val_loss = evaluate_model(
        model, make_loader(), make_loader(), torch.device("cpu"), eval_iter=1
    )

    assert math.isfinite(train_loss)
    assert math.isfinite(val_loss)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_generate_and_print_sample_restores_train_mode(capsys):
    model = TinyLanguageModel()
    model.train()

    generate_and_print_sample(model, DummyTokenizer(), torch.device("cpu"), "prompt")

    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert "Generated text:" in capsys.readouterr().out


def test_train_model_simple_counts_tokens_and_updates_parameters():
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial_logits = model.logits.detach().clone()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=make_loader(),
        val_loader=make_loader(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        num_epochs=2,
        eval_freq=1,
        eval_iter=1,
        start_context="prompt",
        tokenizer=DummyTokenizer(),
    )

    assert tokens_seen == [2, 4]
    assert len(train_losses) == len(val_losses) == 2
    assert train_losses[-1] < train_losses[0]
    assert not torch.equal(model.logits.detach(), initial_logits)
