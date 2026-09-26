"""Smoke script for MultiHeadAttentionStacked and MultiHeadAttentionCombined.

Run from repo root:
    python3 scripts/multi_head_smoke.py
"""

import torch

from sparklm.attention.multi_head import MultiHeadAttentionCombined, MultiHeadAttentionStacked


def main():
    torch.manual_seed(123)
    torch.set_printoptions(precision=4, sci_mode=False)

    x = torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
            ]
        ],
        dtype=torch.float32,
    )

    print("input shape:", tuple(x.shape))

    stacked = MultiHeadAttentionStacked(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    stacked.eval()
    stacked_out = stacked(x)
    print("\n=== stacked multi-head attention ===")
    print("output shape:", tuple(stacked_out.shape))
    print(stacked_out)

    combined = MultiHeadAttentionCombined(
        d_in=3,
        d_out=4,
        context_length=4,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False,
    )
    combined.eval()
    combined_out = combined(x)
    print("\n=== combined multi-head attention ===")
    print("output shape:", tuple(combined_out.shape))
    print(combined_out)


if __name__ == "__main__":
    main()
