import pytest
import torch

from sparklm.attention.multi_head import MultiHeadAttentionCombined, MultiHeadAttentionStacked


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


def test_multi_head_stacked_output_shape():
    torch.manual_seed(123)
    x = _example_batch()
    module = MultiHeadAttentionStacked(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    module.eval()

    out = module(x)

    assert out.shape == torch.Size([2, 4, 4])
    assert torch.isfinite(out).all()


def test_multi_head_combined_output_shape():
    torch.manual_seed(123)
    x = _example_batch()
    module = MultiHeadAttentionCombined(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    module.eval()

    out = module(x)

    assert out.shape == torch.Size([2, 4, 4])
    assert torch.isfinite(out).all()


def test_multi_head_combined_future_token_does_not_change_earlier_outputs():
    torch.manual_seed(123)
    x = _example_batch()
    x_changed = x.clone()
    x_changed[:, 3, :] = torch.tensor([9.0, 9.0, 9.0])

    module = MultiHeadAttentionCombined(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    module.eval()

    out = module(x)
    out_changed = module(x_changed)

    assert torch.allclose(out[:, :3, :], out_changed[:, :3, :], atol=1e-6)


def test_multi_head_stacked_future_token_does_not_change_earlier_outputs():
    torch.manual_seed(123)
    x = _example_batch()
    x_changed = x.clone()
    x_changed[:, 3, :] = torch.tensor([9.0, 9.0, 9.0])

    module = MultiHeadAttentionStacked(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    module.eval()

    out = module(x)
    out_changed = module(x_changed)

    assert torch.allclose(out[:, :3, :], out_changed[:, :3, :], atol=1e-6)


def test_multi_head_combined_rejects_too_long_sequence():
    x = torch.randn(1, 5, 3)
    module = MultiHeadAttentionCombined(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )

    with pytest.raises(ValueError, match="exceeds maximum"):
        module(x)


def test_multi_head_combined_parameters_receive_gradients():
    torch.manual_seed(123)
    x = _example_batch()
    module = MultiHeadAttentionCombined(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )

    loss = module(x).sum()
    loss.backward()

    assert module.W_query.weight.grad is not None
    assert module.W_key.weight.grad is not None
    assert module.W_value.weight.grad is not None
    assert module.out_proj.weight.grad is not None
    assert torch.isfinite(module.W_query.weight.grad).all()
    assert torch.isfinite(module.W_key.weight.grad).all()
    assert torch.isfinite(module.W_value.weight.grad).all()
    assert torch.isfinite(module.out_proj.weight.grad).all()
