import pytest
import torch

from sparklm.model import GPTModel


def tiny_config(**overrides):
    config = {
        "vocab_size": 31,
        "context_length": 8,
        "embedding_dim": 12,
        "num_heads": 3,
        "num_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    config.update(overrides)
    return config


def test_gpt_model_returns_logits_for_each_token():
    model = GPTModel(tiny_config()).eval()
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits = model(token_ids)

    assert logits.shape == (2, 4, 31)
    assert torch.isfinite(logits).all()
    assert len(model.transformer_blocks) == 2


def test_gpt_model_prevents_future_token_leakage():
    torch.manual_seed(123)
    model = GPTModel(tiny_config()).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])
    changed_token_ids = token_ids.clone()
    changed_token_ids[:, -1] = 5

    with torch.no_grad():
        logits = model(token_ids)
        changed_logits = model(changed_token_ids)

    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)


def test_gpt_model_backward_produces_finite_gradients():
    torch.manual_seed(123)
    model = GPTModel(tiny_config())
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    model(token_ids).square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gpt_model_rejects_sequences_longer_than_context():
    model = GPTModel(tiny_config(context_length=4))
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with pytest.raises(ValueError, match="exceeds context length"):
        model(token_ids)
