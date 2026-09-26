"""Smoke script for ScaledDotProductAttention.

Run from repo root:
    python3 scripts/scaled_dot_product_smoke.py
"""

import torch

from sparklm.attention.scaled_dot_product import ScaledDotProductAttention


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    attn = ScaledDotProductAttention(d_in=2)
    context, weights = attn(q, k, v)

    print("=== scaled dot-product attention ===")
    print("q shape:", tuple(q.shape))
    print("k shape:", tuple(k.shape))
    print("v shape:", tuple(v.shape))
    print("weights shape:", tuple(weights.shape))
    print("weights:")
    print(weights)
    print("weights row sums:", weights.sum(dim=-1))
    print("context shape:", tuple(context.shape))
    print("context:")
    print(context)

    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    masked_context, masked_weights = attn(q, k, v, mask=mask)

    print("\n=== masked attention ===")
    print("mask:")
    print(mask)
    print("masked weights:")
    print(masked_weights)
    print("masked weights row sums:", masked_weights.sum(dim=-1))
    print("masked context:")
    print(masked_context)


if __name__ == "__main__":
    main()
