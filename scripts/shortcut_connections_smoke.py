"""Compare gradient flow with and without shortcut connections.

From the repository root:
    .venv/bin/python scripts/shortcut_connections_smoke.py
"""

import torch
from torch import nn

from sparklm.example_deep_network import ExampleDeepNetwork


def mean_weight_gradients(model):
    return [layer[0].weight.grad.abs().mean().item() for layer in model.layers]


def main():
    torch.manual_seed(123)
    layer_sizes = [3, 3, 3, 3, 3, 1]
    inputs = torch.tensor([[1.0, 0.0, -1.0]])
    target = torch.tensor([[0.5]])

    plain_model = ExampleDeepNetwork(layer_sizes, use_shortcut=False)
    shortcut_model = ExampleDeepNetwork(layer_sizes, use_shortcut=True)
    shortcut_model.load_state_dict(plain_model.state_dict())

    plain_loss = nn.functional.mse_loss(plain_model(inputs), target)
    shortcut_loss = nn.functional.mse_loss(shortcut_model(inputs), target)
    plain_loss.backward()
    shortcut_loss.backward()

    print("layer sizes:", layer_sizes)
    print("plain loss:", plain_loss.item())
    print("shortcut loss:", shortcut_loss.item())
    print("plain mean absolute weight gradients:", mean_weight_gradients(plain_model))
    print("shortcut mean absolute weight gradients:", mean_weight_gradients(shortcut_model))


if __name__ == "__main__":
    main()
