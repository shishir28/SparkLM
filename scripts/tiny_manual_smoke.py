"""Smoke script focused on tiny_manual_attention_example.

Run from repo root:
    python scripts/tiny_manual_smoke.py

This script calls tiny_manual_attention_example (canonical and a small
custom example) and prints only the essential intermediate tensors.
"""

import torch
from sparklm.attention.tiny_manual import tiny_manual_attention_example


def main():
    torch.set_printoptions(precision=6, sci_mode=False)

    print("=== canonical tiny manual attention example ===")
    out = tiny_manual_attention_example()
    # accept either lowercase or uppercase 'q'
    q_key = out.get("q", out.get("Q"))
    K_key = out.get("K", out.get("k"))
    V_key = out.get("V", out.get("v"))

    print("q:", q_key)
    print("K:", K_key)
    print("V:", V_key)
    print("weights_loop:", out["weights_loop"])
    print("context_loop:", out["context_loop"])

    print("\n=== custom small example ===")
    q = torch.tensor([2.0, 0.0])
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out2 = tiny_manual_attention_example(q=q, k=K, v=V)
    print("weights_loop:", out2["weights_loop"])
    print("context_loop:", out2["context_loop"])


if __name__ == "__main__":
    main()
