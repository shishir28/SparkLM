import torch
from torch import nn

from sparklm.layers import LayerNorm


def test_layer_norm_preserves_shape_and_normalizes_last_dimension():
    torch.manual_seed(123)
    inputs = torch.randn(2, 3, 4)

    outputs = LayerNorm(4)(inputs)

    assert outputs.shape == inputs.shape
    assert torch.allclose(outputs.mean(dim=-1), torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(outputs.var(dim=-1, unbiased=False), torch.ones(2, 3), atol=1e-4)


def test_layer_norm_matches_pytorch_reference():
    inputs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            [[-1.0, 0.0, 1.0, 2.0], [3.0, 3.5, 4.0, 5.0]],
        ]
    )
    layer = LayerNorm(4)
    reference = nn.LayerNorm(4, eps=layer.eps)

    with torch.no_grad():
        layer.scale.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        layer.shift.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        reference.weight.copy_(layer.scale)
        reference.bias.copy_(layer.shift)

    assert torch.allclose(layer(inputs), reference(inputs), atol=1e-6)


def test_layer_norm_handles_constant_inputs():
    layer = LayerNorm(4)
    inputs = torch.full((2, 3, 4), 7.0)

    outputs = layer(inputs)

    assert torch.isfinite(outputs).all()
    assert torch.equal(outputs, torch.zeros_like(inputs))


def test_layer_norm_backward_produces_finite_gradients():
    torch.manual_seed(123)
    layer = LayerNorm(4)
    inputs = torch.randn(2, 3, 4, requires_grad=True)

    layer(inputs).square().sum().backward()

    for gradient in (inputs.grad, layer.scale.grad, layer.shift.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
