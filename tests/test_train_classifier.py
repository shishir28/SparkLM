import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sparklm.classification.classifier_model import add_classification_head
from sparklm.classification.metrics import calc_loss_loader
from sparklm.classification.train_classifier import (
    evaluate_model,
    train_classifier_simple,
)
from sparklm.model import GPTModel


TINY_CONFIG = {
    "vocab_size": 16,
    "context_length": 4,
    "embedding_dim": 32,
    "num_layers": 2,
    "num_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def make_classifier():
    torch.manual_seed(0)
    return add_classification_head(GPTModel(TINY_CONFIG))


def make_dataset(num_examples=8):
    # The label is the final token (0 or 1); earlier tokens are noise.
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(
        2, TINY_CONFIG["vocab_size"], (num_examples, 4), generator=generator
    )
    labels = torch.arange(num_examples) % 2
    inputs[:, -1] = labels
    return TensorDataset(inputs, labels)


def make_loader(batch_size=4, shuffle=False, num_examples=8):
    return DataLoader(
        make_dataset(num_examples), batch_size=batch_size, shuffle=shuffle
    )


def make_optimizer(model, lr=1e-2):
    # Same call shape as the script: frozen parameters are passed but get no grads.
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)


def train(model, loader, num_epochs=1, eval_freq=1, eval_iter=None):
    return train_classifier_simple(
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=make_optimizer(model),
        device="cpu",
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=len(loader) if eval_iter is None else eval_iter,
    )


def snapshot(model, trainable):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad is trainable
    }


def test_training_changes_trainable_and_keeps_frozen_parameters():
    model = make_classifier()
    frozen_before = snapshot(model, trainable=False)
    trainable_before = snapshot(model, trainable=True)

    train(model, make_loader(batch_size=8))

    current = dict(model.named_parameters())
    assert all(
        torch.equal(current[name], value) for name, value in frozen_before.items()
    )
    assert any(
        not torch.equal(current[name], value)
        for name, value in trainable_before.items()
    )


def test_tiny_dataset_reaches_perfect_training_accuracy():
    torch.manual_seed(123)
    model = make_classifier()

    _, _, train_accs, val_accs, _ = train(
        model, make_loader(shuffle=True), num_epochs=30, eval_freq=10
    )

    assert train_accs[-1] == 1.0
    assert val_accs[-1] == 1.0


@pytest.mark.parametrize(
    ("batch_size", "num_epochs", "eval_freq"),
    [(4, 3, 1), (4, 3, 2), (3, 2, 4)],
)
def test_history_lengths_match_recording_schedule(batch_size, num_epochs, eval_freq):
    loader = make_loader(batch_size=batch_size)
    total_steps = len(loader) * num_epochs

    train_losses, val_losses, train_accs, val_accs, _ = train(
        make_classifier(), loader, num_epochs=num_epochs, eval_freq=eval_freq
    )

    # Steps are numbered from 0, and step 0 is always evaluated.
    expected_loss_points = (total_steps - 1) // eval_freq + 1
    assert len(train_losses) == len(val_losses) == expected_loss_points
    assert len(train_accs) == len(val_accs) == num_epochs


def test_examples_seen_sums_batch_sizes_including_partial_batch():
    # 8 examples with batch size 3 give batches of 3, 3 and 2.
    loader = make_loader(batch_size=3)

    *_, examples_seen = train(make_classifier(), loader, num_epochs=2)

    assert examples_seen == 16


def test_training_steps_use_train_mode_and_evaluation_uses_eval_mode():
    model = make_classifier()
    modes = []
    model.register_forward_pre_hook(
        lambda module, inputs: modes.append((torch.is_grad_enabled(), module.training))
    )

    train(model, make_loader(), num_epochs=2, eval_freq=1)

    grad_modes = {training for grad_enabled, training in modes if grad_enabled}
    no_grad_modes = {training for grad_enabled, training in modes if not grad_enabled}
    assert grad_modes == {True}
    assert no_grad_modes == {False}


def test_training_returns_model_in_train_mode():
    model = make_classifier()

    train(model, make_loader(), num_epochs=1)

    assert model.training


def test_evaluate_model_matches_loader_loss_and_restores_train_mode():
    model = make_classifier()
    loader = make_loader()
    model.eval()
    expected = calc_loss_loader(loader, model, "cpu", num_batches=1)
    model.train()

    train_loss, val_loss = evaluate_model(model, loader, loader, "cpu", eval_iter=1)

    assert train_loss == pytest.approx(expected)
    assert val_loss == pytest.approx(expected)
    assert model.training
