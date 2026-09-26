"""Smoke script for SelfAttentionLinear_v1 and SelfAttentionLinear_v2.

Run from repo root:
    python3 scripts/self_attention_linear_smoke.py
"""

import torch

from sparklm.attention.self_attention_linear import SelfAttentionLinear_v1, SelfAttentionLinear_v2


def main():
    torch.manual_seed(123)
    torch.set_printoptions(precision=4, sci_mode=False)

    x = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        dtype=torch.float32,
    )

    print("input shape:", tuple(x.shape))

    v1 = SelfAttentionLinear_v1(d_in=3, d_out=2)
    out_v1 = v1(x)
    print("v1 output shape:", tuple(out_v1.shape))
    print("v1 output:")
    print(out_v1)

    v2 = SelfAttentionLinear_v2(d_in=3, d_out=2, qkv_bias=False)
    out_v2 = v2(x)
    print("v2 output shape:", tuple(out_v2.shape))
    print("v2 output:")
    print(out_v2)

    with torch.no_grad():
        v2.W_query.weight.copy_(v1.W_query.T)
        v2.W_key.weight.copy_(v1.W_key.T)
        v2.W_value.weight.copy_(v1.W_value.T)

    aligned_out_v2 = v2(x)
    print("v1 and weight-aligned v2 match:", torch.allclose(out_v1, aligned_out_v2, atol=1e-6))


if __name__ == "__main__":
    main()
