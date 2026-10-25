import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sparklm.classification.metrics import (
    calc_accuracy_loader,
    calc_loss_batch,
    calc_loss_loader,
)


# Rows are indexed by the final token id of each input sequence.
LAST_POSITION_LOGITS = torch.tensor(
    [
        [2.0, 0.0],
        [0.0, 1.0],
        [3.0, 0.0],
        [0.0, 2.0],
    ]
)
# Argmax predicts [0, 1, 0, 1]; only the last example is wrong.
LABELS = torch.tensor([0, 1, 0, 0])


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, earlier_value=0.0):
        super().__init__()
        self.earlier_value = earlier_value
        self.offset = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        logits = torch.full((batch_size, seq_length, 2), self.earlier_value)
        logits[:, -1, :] = LAST_POSITION_LOGITS[input_ids[:, -1]]
        return logits + self.offset


def make_inputs(earlier_token=7):
    example_ids = torch.arange(len(LABELS))
    return torch.stack(
        [torch.full_like(example_ids, earlier_token), example_ids], dim=1
    )


def make_loader(batch_size=2, inputs=None, labels=LABELS):
    inputs = make_inputs() if inputs is None else inputs
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def make_empty_loader():
    return make_loader(inputs=make_inputs()[:0], labels=LABELS[:0])


def two_class_cross_entropy(logits, label):
    return -logits[label] + math.log(math.exp(logits[0]) + math.exp(logits[1]))


def expected_mean_loss(indices):
    return sum(
        two_class_cross_entropy(LAST_POSITION_LOGITS[i].tolist(), LABELS[i].item())
        for i in indices
    ) / len(indices)


def test_accuracy_matches_hand_count():
    assert calc_accuracy_loader(make_loader(), FixedLogitsModel(), "cpu") == 0.75


def test_batch_loss_matches_hand_computed_cross_entropy():
    loss = calc_loss_batch(make_inputs(), LABELS, FixedLogitsModel(), "cpu")

    assert loss.item() == pytest.approx(expected_mean_loss(range(4)))


def test_loader_loss_is_mean_of_batch_losses():
    loss = calc_loss_loader(make_loader(batch_size=2), FixedLogitsModel(), "cpu")

    batch_means = (expected_mean_loss([0, 1]), expected_mean_loss([2, 3]))
    assert loss == pytest.approx(sum(batch_means) / 2)


def test_earlier_positions_do_not_affect_metrics():
    loader = make_loader()
    reference = FixedLogitsModel(earlier_value=0.0)
    changed = FixedLogitsModel(earlier_value=100.0)

    assert calc_accuracy_loader(loader, changed, "cpu") == calc_accuracy_loader(
        loader, reference, "cpu"
    )
    assert calc_loss_loader(loader, changed, "cpu") == pytest.approx(
        calc_loss_loader(loader, reference, "cpu")
    )


def test_earlier_tokens_do_not_affect_metrics():
    loader = make_loader(inputs=make_inputs(earlier_token=3))

    assert calc_accuracy_loader(loader, FixedLogitsModel(), "cpu") == 0.75


@pytest.mark.parametrize(
    ("num_batches", "expected_accuracy"),
    [(1, 1.0), (2, 0.75), (99, 0.75)],
)
def test_accuracy_num_batches_limits_and_caps(num_batches, expected_accuracy):
    accuracy = calc_accuracy_loader(
        make_loader(), FixedLogitsModel(), "cpu", num_batches=num_batches
    )

    assert accuracy == expected_accuracy


@pytest.mark.parametrize(
    ("num_batches", "expected_indices"),
    [(1, [0, 1]), (99, range(4))],
)
def test_loss_num_batches_limits_and_caps(num_batches, expected_indices):
    loss = calc_loss_loader(
        make_loader(), FixedLogitsModel(), "cpu", num_batches=num_batches
    )

    assert loss == pytest.approx(expected_mean_loss(list(expected_indices)))


@pytest.mark.parametrize("metric", [calc_accuracy_loader, calc_loss_loader])
def test_empty_loader_returns_nan(metric):
    assert math.isnan(metric(make_empty_loader(), FixedLogitsModel(), "cpu"))


@pytest.mark.parametrize("metric", [calc_accuracy_loader, calc_loss_loader])
def test_zero_batches_returns_nan(metric):
    result = metric(make_loader(), FixedLogitsModel(), "cpu", num_batches=0)

    assert math.isnan(result)


def test_accuracy_does_not_create_gradients():
    model = FixedLogitsModel()

    calc_accuracy_loader(make_loader(), model, "cpu")

    assert model.offset.grad is None


def test_batch_loss_keeps_train_mode_and_is_differentiable():
    model = FixedLogitsModel()
    model.train()

    loss = calc_loss_batch(make_inputs(), LABELS, model, "cpu")
    loss.backward()

    assert model.training
    assert model.offset.grad is not None
