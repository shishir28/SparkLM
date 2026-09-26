import pytest
import torch

from sparklm.attention.causal_self_attention import CausalSelfAttention


def _example_batch():
    return torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
            ],
            [
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
                [0.31, 0.44, 0.72],
                [0.92, 0.12, 0.33],
            ],
        ],
        dtype=torch.float32,
    )


def _manual_attention_weights(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    keys = module.W_key(x)
    queries = module.W_query(x)
    num_tokens = x.shape[1]
    scores = queries @ keys.transpose(1, 2)
    mask = module.causal_mask[:num_tokens, :num_tokens].bool()
    scores = scores.masked_fill(mask, -torch.inf)
    return torch.softmax(scores / (keys.shape[-1] ** 0.5), dim=-1)


def test_causal_self_attention_output_shape():
    torch.manual_seed(123)
    x = _example_batch()
    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)
    module.eval()

    out = module(x)

    assert out.shape == torch.Size([2, 4, 2])
    assert torch.isfinite(out).all()


def test_causal_self_attention_future_weights_are_zero():
    torch.manual_seed(123)
    x = _example_batch()
    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)
    module.eval()

    weights = _manual_attention_weights(module, x)
    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.allclose(weights[:, future_mask], torch.zeros_like(weights[:, future_mask]), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_causal_self_attention_manual_formula_matches_forward():
    torch.manual_seed(123)
    x = _example_batch()
    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)
    module.eval()

    weights = _manual_attention_weights(module, x)
    values = module.W_value(x)
    expected = weights @ values

    assert torch.allclose(module(x), expected, atol=1e-6)


def test_causal_self_attention_future_token_does_not_change_earlier_outputs():
    torch.manual_seed(123)
    x = _example_batch()
    x_changed = x.clone()
    x_changed[:, 3, :] = torch.tensor([9.0, 9.0, 9.0])

    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)
    module.eval()

    out = module(x)
    out_changed = module(x_changed)

    assert torch.allclose(out[:, :3, :], out_changed[:, :3, :], atol=1e-6)


def test_causal_self_attention_rejects_too_long_sequence():
    x = torch.randn(1, 5, 3)
    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)

    with pytest.raises(ValueError, match="exceeds maximum"):
        module(x)


def test_causal_self_attention_parameters_receive_gradients():
    torch.manual_seed(123)
    x = _example_batch()
    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)

    loss = module(x).sum()
    loss.backward()

    assert module.W_query.weight.grad is not None
    assert module.W_key.weight.grad is not None
    assert module.W_value.weight.grad is not None
    assert torch.isfinite(module.W_query.weight.grad).all()
    assert torch.isfinite(module.W_key.weight.grad).all()
    assert torch.isfinite(module.W_value.weight.grad).all()
