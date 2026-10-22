import torch
import torch.nn.functional as F

from sparklm.checkpoint import load_checkpoint, save_checkpoint
from sparklm.model import GPTModel


TINY_CONFIG = {
    "vocab_size": 11,
    "context_length": 4,
    "embedding_dim": 8,
    "num_layers": 1,
    "num_heads": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
INPUTS = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
TARGETS = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])


def train_step(model, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(INPUTS)
    loss = F.cross_entropy(logits.flatten(0, 1), TARGETS.flatten())
    loss.backward()
    optimizer.step()
    return loss.detach()


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            nested_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def make_model_and_optimizer():
    model = GPTModel(TINY_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    return model, optimizer


def test_checkpoint_restores_logits_loss_optimizer_and_metadata(tmp_path):
    torch.manual_seed(123)
    model, optimizer = make_model_and_optimizer()
    train_step(model, optimizer)
    metadata = {"seed": 123, "global_step": 1, "tokens_seen": INPUTS.numel()}
    checkpoint_path = tmp_path / "checkpoint.pt"

    model.eval()
    with torch.no_grad():
        expected_logits = model(INPUTS)
        expected_loss = F.cross_entropy(
            expected_logits.flatten(0, 1), TARGETS.flatten()
        )
    save_checkpoint(checkpoint_path, model, optimizer, TINY_CONFIG, metadata)

    restored_model, restored_optimizer = make_model_and_optimizer()
    restored_metadata = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )
    restored_model.eval()
    with torch.no_grad():
        restored_logits = restored_model(INPUTS)
        restored_loss = F.cross_entropy(
            restored_logits.flatten(0, 1), TARGETS.flatten()
        )

    assert torch.equal(restored_logits, expected_logits)
    assert torch.equal(restored_loss, expected_loss)
    assert nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert restored_metadata == metadata

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["config"] == TINY_CONFIG


def test_checkpoint_resume_matches_uninterrupted_next_step(tmp_path):
    torch.manual_seed(123)
    model, optimizer = make_model_and_optimizer()
    train_step(model, optimizer)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, TINY_CONFIG, {})

    restored_model, restored_optimizer = make_model_and_optimizer()
    load_checkpoint(checkpoint_path, restored_model, restored_optimizer)

    expected_loss = train_step(model, optimizer)
    restored_loss = train_step(restored_model, restored_optimizer)

    assert torch.equal(restored_loss, expected_loss)
    assert nested_equal(restored_model.state_dict(), model.state_dict())
    assert nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
