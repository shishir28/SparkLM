import pytest
import torch

from sparklm.example_deep_network import ExampleDeepNetwork


def zero_linear_layers(model):
    with torch.no_grad():
        for layer in model.layers:
            layer[0].weight.zero_()
            layer[0].bias.zero_()


def test_deep_network_returns_expected_shape():
    model = ExampleDeepNetwork([3, 4, 2], use_shortcut=False)
    inputs = torch.randn(5, 3)

    outputs = model(inputs)

    assert outputs.shape == (5, 2)
    assert torch.isfinite(outputs).all()


def test_shortcut_returns_input_when_equal_width_branches_are_zero():
    model = ExampleDeepNetwork([3, 3, 3], use_shortcut=True)
    zero_linear_layers(model)
    inputs = torch.randn(2, 3)

    outputs = model(inputs)

    assert torch.equal(outputs, inputs)


def test_shortcut_is_skipped_when_dimensions_change():
    model = ExampleDeepNetwork([3, 2], use_shortcut=True)
    zero_linear_layers(model)
    inputs = torch.randn(2, 3)

    outputs = model(inputs)

    assert torch.equal(outputs, torch.zeros(2, 2))


@pytest.mark.parametrize("use_shortcut", [False, True])
def test_deep_network_backward_produces_finite_gradients(use_shortcut):
    torch.manual_seed(123)
    model = ExampleDeepNetwork([3, 3, 3, 1], use_shortcut=use_shortcut)
    inputs = torch.randn(2, 3, requires_grad=True)

    model(inputs).square().sum().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
