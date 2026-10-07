"""Run the feed-forward network independently.

From the repository root:
    .venv/bin/python scripts/feed_forward_smoke.py
"""

import torch

from sparklm.layers import FeedForward


def main():
    torch.manual_seed(123)
    layer = FeedForward({"emb_dim": 8})
    inputs = torch.randn(2, 3, 8, requires_grad=True)

    outputs = layer(inputs)
    outputs.square().mean().backward()

    print("input shape:", tuple(inputs.shape))
    print("expanded shape:", tuple(layer.layers[0](inputs).shape))
    print("output shape:", tuple(outputs.shape))
    print("output is finite:", torch.isfinite(outputs).all().item())
    print("input gradient is finite:", torch.isfinite(inputs.grad).all().item())
    print("parameter count:", sum(parameter.numel() for parameter in layer.parameters()))


if __name__ == "__main__":
    main()
