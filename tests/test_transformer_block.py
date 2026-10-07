import torch

from sparklm.transformer_block import TransformerBlock


def tiny_config():
    return {
        "embedding_dim": 8,
        "context_length": 6,
        "num_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }


def test_transformer_block_preserves_shape():
    block = TransformerBlock(tiny_config()).eval()
    inputs = torch.randn(2, 4, 8)

    outputs = block(inputs)

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_transformer_block_prevents_future_token_leakage():
    torch.manual_seed(123)
    block = TransformerBlock(tiny_config()).eval()
    inputs = torch.randn(1, 4, 8)
    changed_inputs = inputs.clone()
    changed_inputs[:, -1] += 10.0

    outputs = block(inputs)
    changed_outputs = block(changed_inputs)

    assert torch.allclose(outputs[:, :-1], changed_outputs[:, :-1], atol=1e-6)


def test_transformer_block_returns_input_when_residual_branches_are_zero():
    block = TransformerBlock(tiny_config()).eval()
    inputs = torch.randn(2, 4, 8)

    with torch.no_grad():
        for parameter in block.att.parameters():
            parameter.zero_()
        for parameter in block.ff.parameters():
            parameter.zero_()

    assert torch.equal(block(inputs), inputs)


def test_transformer_block_backward_produces_finite_gradients():
    torch.manual_seed(123)
    block = TransformerBlock(tiny_config())
    inputs = torch.randn(2, 4, 8, requires_grad=True)

    block(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
