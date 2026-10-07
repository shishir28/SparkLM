"""Compare the manual GELU with PyTorch's tanh approximation.

From the repository root:
    .venv/bin/python scripts/gelu_smoke.py
"""

import torch
from torch import nn

from sparklm.layers import GELU


def main():
    inputs = torch.linspace(-3.0, 3.0, 13)

    with torch.no_grad():
        manual_output = GELU()(inputs)
        reference_output = nn.functional.gelu(inputs, approximate="tanh")

    print("inputs:", inputs)
    print("manual GELU:", manual_output)
    print("PyTorch GELU:", reference_output)
    print("matches PyTorch GELU:", torch.allclose(manual_output, reference_output, atol=1e-6))
    print("maximum absolute difference:", (manual_output - reference_output).abs().max().item())


if __name__ == "__main__":
    main()
