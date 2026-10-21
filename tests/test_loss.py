import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sparklm.data_loader import create_dataloader_v1
from sparklm.loss import calc_loss_batch, calc_loss_loader


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.calls = 0

    def forward(self, input_batch):
        self.calls += 1
        batch_size, sequence_length = input_batch.shape
        return self.fixed_logits.expand(batch_size, sequence_length, -1)


def test_calc_loss_batch_matches_hand_computed_cross_entropy():
    inputs = torch.tensor([[0, 1]])
    targets = torch.tensor([[0, 1]])
    model = FixedLogitModel([2.0, 0.0])

    actual = calc_loss_batch(inputs, targets, model, torch.device("cpu"))
    probability_of_class_0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    probability_of_class_1 = 1.0 / (math.exp(2.0) + 1.0)
    expected = torch.tensor(
        -(math.log(probability_of_class_0) + math.log(probability_of_class_1)) / 2
    )

    assert torch.allclose(actual, expected)


def test_calc_loss_batch_flattens_all_batch_positions():
    inputs = torch.tensor([[0, 1], [2, 3]])
    targets = torch.tensor([[0, 1], [1, 0]])
    model = FixedLogitModel([2.0, 0.0])

    actual = calc_loss_batch(inputs, targets, model, torch.device("cpu"))
    logits = model.fixed_logits.expand(4, -1)
    expected = torch.nn.functional.cross_entropy(logits, targets.flatten())

    assert actual.shape == torch.Size([])
    assert torch.allclose(actual, expected)


def test_calc_loss_loader_limits_number_of_batches():
    inputs = torch.arange(3).reshape(3, 1)
    targets = torch.tensor([[0], [1], [0]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)
    model = FixedLogitModel([2.0, 0.0])

    actual = calc_loss_loader(loader, model, torch.device("cpu"), num_batches=2)
    first_two = [
        calc_loss_batch(inputs[i : i + 1], targets[i : i + 1], model, torch.device("cpu"))
        for i in range(2)
    ]

    assert math.isclose(actual, torch.stack(first_two).mean().item(), rel_tol=1e-6)
    assert model.calls == 4


def test_calc_loss_loader_returns_nan_for_empty_loader():
    empty_dataset = TensorDataset(torch.empty((0, 1), dtype=torch.long), torch.empty((0, 1), dtype=torch.long))
    loader = DataLoader(empty_dataset, batch_size=1)
    model = FixedLogitModel([2.0, 0.0])

    result = calc_loss_loader(loader, model, torch.device("cpu"))

    assert math.isnan(result)
    assert model.calls == 0


def test_training_loader_does_not_cross_validation_split():
    class CharacterTokenizer:
        def encode(self, text):
            return [ord(character) for character in text]

    train_text = "abcdef"
    validation_text = "ghij"
    loader = create_dataloader_v1(
        train_text,
        tokenizer=CharacterTokenizer(),
        batch_size=1,
        max_length=3,
        stride=3,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    train_token_ids = {ord(character) for character in train_text}
    validation_token_ids = {ord(character) for character in validation_text}
    for input_batch, target_batch in loader:
        assert set(input_batch.flatten().tolist()).issubset(train_token_ids)
        assert set(target_batch.flatten().tolist()).issubset(train_token_ids)
        assert not set(input_batch.flatten().tolist()) & validation_token_ids
        assert not set(target_batch.flatten().tolist()) & validation_token_ids
