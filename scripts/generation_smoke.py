"""Generate text greedily with the untrained GPT model.

From the repository root:
    .venv/bin/python scripts/generation_smoke.py
"""

import torch

from sparklm.generation import generate_text_simple
from sparklm.model import GPTModel
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main():
    torch.manual_seed(123)
    tokenizer = BPETokenizer()
    prompt = "Every effort moves you"
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    config = {
        "vocab_size": tokenizer.vocab_size(),
        "context_length": 8,
        "embedding_dim": 12,
        "num_heads": 3,
        "num_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(config).eval()

    generated_ids = generate_text_simple(
        model,
        token_ids,
        max_new_tokens=6,
        context_size=config["context_length"],
    )

    print("prompt:", prompt)
    print("prompt token count:", token_ids.shape[1])
    print("generated shape:", tuple(generated_ids.shape))
    print("generated token IDs:", generated_ids)
    print("generated text:", tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
