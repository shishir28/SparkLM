"""Run a Transformer block forward and backward.

From the repository root:
    .venv/bin/python scripts/transformer_block_smoke.py
"""

import torch

from sparklm.transformer_block import TransformerBlock


def main():
    torch.manual_seed(123)
    config = {
        "embedding_dim": 8,
        "context_length": 6,
        "num_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    block = TransformerBlock(config).eval()
    inputs = torch.randn(2, 4, 8, requires_grad=True)

    outputs = block(inputs)
    outputs.square().mean().backward()

    changed_inputs = inputs.detach().clone()
    changed_inputs[:, -1] += 10.0
    with torch.no_grad():
        changed_outputs = block(changed_inputs)

    print("input shape:", tuple(inputs.shape))
    print("output shape:", tuple(outputs.shape))
    print("output is finite:", torch.isfinite(outputs).all().item())
    print("input gradient is finite:", torch.isfinite(inputs.grad).all().item())
    print(
        "earlier positions ignore changed future token:",
        torch.allclose(outputs[:, :-1], changed_outputs[:, :-1], atol=1e-6),
    )


if __name__ == "__main__":
    main()
