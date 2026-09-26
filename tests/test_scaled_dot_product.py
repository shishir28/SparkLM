import pytest
import torch

from sparklm.attention.scaled_dot_product import ScaledDotProductAttention


def test_scaled_dot_product_output_and_weight_shapes():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attn = ScaledDotProductAttention(d_in=2)

    context, weights = attn(q, k, v)

    assert context.shape == torch.Size([2, 2])
    assert weights.shape == torch.Size([2, 3])
    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()


def test_scaled_dot_product_weights_rows_sum_to_one_without_dropout():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attn = ScaledDotProductAttention(d_in=2)

    _, weights = attn(q, k, v)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_scaled_dot_product_matches_manual_formula():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attn = ScaledDotProductAttention(d_in=2)

    context, weights = attn(q, k, v)

    expected_scores = q @ k.transpose(-2, -1)
    expected_scores = expected_scores / (q.shape[-1] ** 0.5)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_context = expected_weights @ v

    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(context, expected_context, atol=1e-6)


def test_scaled_dot_product_mask_blocks_positions():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    attn = ScaledDotProductAttention(d_in=2)

    _, weights = attn(q, k, v, mask=mask)

    assert torch.isclose(weights[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_scaled_dot_product_batched_input_shapes():
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    attn = ScaledDotProductAttention(d_in=4)

    context, weights = attn(q, k, v)

    assert context.shape == torch.Size([2, 3, 6])
    assert weights.shape == torch.Size([2, 3, 5])
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_scaled_dot_product_rejects_invalid_shapes():
    attn = ScaledDotProductAttention(d_in=2)

    with pytest.raises(ValueError, match="at least 2D"):
        attn(torch.tensor([1.0, 2.0]), torch.randn(2, 2), torch.randn(2, 2))

    with pytest.raises(ValueError, match="same final dimension"):
        attn(torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 5))

    with pytest.raises(ValueError, match="same token/key length"):
        attn(torch.randn(2, 3), torch.randn(4, 3), torch.randn(5, 2))
