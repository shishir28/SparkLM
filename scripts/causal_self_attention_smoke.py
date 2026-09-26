"""Smoke script for CausalSelfAttention.

Run from repo root:
    python3 scripts/causal_self_attention_smoke.py
"""

import torch

from sparklm.attention.causal_self_attention import CausalSelfAttention


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

    module = CausalSelfAttention(d_in=3, d_out=2, max_seq_len=4, attention_dropout=0.0)
    module.eval()

    out = module(x)

    print("=== causal self-attention ===")
    print("input shape:", tuple(x.shape))
    print("causal mask:")
    print(module.causal_mask[: x.shape[1], : x.shape[1]])
    print("output shape:", tuple(out.shape))
    print("output:")
    print(out)

    keys = module.W_key(x)
    queries = module.W_query(x)
    scores = queries @ keys.transpose(1, 2)
    scores = scores.masked_fill(module.causal_mask[: x.shape[1], : x.shape[1]].bool(), -torch.inf)
    weights = torch.softmax(scores / (keys.shape[-1] ** 0.5), dim=-1)

    print("attention weights:")
    print(weights)
    print("attention weight row sums:")
    print(weights.sum(dim=-1))


if __name__ == "__main__":
    main()
