"""Compare the manual LayerNorm with PyTorch's implementation.

From the repository root:
    .venv/bin/python scripts/layer_norm_smoke.py
"""

import torch
from torch import nn

from sparklm.layers import LayerNorm


def main():
    inputs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            [[-1.0, 0.0, 1.0, 2.0], [3.0, 3.5, 4.0, 5.0]],
        ]
    )
    layer = LayerNorm(4)
    reference = nn.LayerNorm(4, eps=layer.eps)

    with torch.no_grad():
        manual_output = layer(inputs)
        reference_output = reference(inputs)

    print("input shape:", tuple(inputs.shape))
    print("output shape:", tuple(manual_output.shape))
    print("output means:", manual_output.mean(dim=-1))
    print("output variances:", manual_output.var(dim=-1, unbiased=False))
    print("matches torch.nn.LayerNorm:", torch.allclose(manual_output, reference_output, atol=1e-6))
    print("maximum absolute difference:", (manual_output - reference_output).abs().max().item())


if __name__ == "__main__":
    main()
