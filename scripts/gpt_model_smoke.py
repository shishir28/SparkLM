"""Run the complete untrained GPT model on GPT-2-tokenized prompts.

From the repository root:
    .venv/bin/python scripts/gpt_model_smoke.py
"""

import torch

from sparklm.model import GPTModel
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main():
    torch.manual_seed(123)
    tokenizer = BPETokenizer()
    prompts = ["Every effort moves you", "Every day holds a"]
    token_ids = torch.tensor(
        [tokenizer.encode(prompt) for prompt in prompts], dtype=torch.long
    )
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

    with torch.no_grad():
        logits = model(token_ids)
        next_token_ids = logits[:, -1].argmax(dim=-1)

    print("prompts:", prompts)
    print("input shape:", tuple(token_ids.shape))
    print("logits shape:", tuple(logits.shape))
    print("parameter count:", sum(parameter.numel() for parameter in model.parameters()))
    print("next token IDs:", next_token_ids)
    print("next tokens:", [tokenizer.decode([token_id]) for token_id in next_token_ids.tolist()])


if __name__ == "__main__":
    main()
