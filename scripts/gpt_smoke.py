"""Run the dummy GPT model on two GPT-2-tokenized prompts.

From the repository root:
    .venv/bin/python scripts/gpt_smoke.py
"""

import torch

from sparklm.dummy_gpt_model import DummyGPTModel
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main():
    torch.manual_seed(123)

    tokenizer = BPETokenizer()
    prompts = ["Every effort moves you", "Every day holds a"]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    token_ids = torch.tensor(encoded_prompts, dtype=torch.long)

    config = {
        "vocab_size": tokenizer.vocab_size(),
        "context_length": 8,
        "embedding_dim": 12,
        "num_layers": 2,
        "drop_rate": 0.1,
    }

   # eval disables dropout and other training-specific behaviors, making the model deterministic for inference.
    model = DummyGPTModel(config).eval()

    # no_grad is used to disable gradient tracking inside its block. Inference uses it to reduce memory and computation.
    with torch.no_grad():
        logits = model(token_ids)

    predicted_ids = logits.argmax(dim=-1)

    print("prompts:", prompts)
    print("token IDs:", token_ids)
    print("input shape:", tuple(token_ids.shape))
    print("logits shape:", tuple(logits.shape))
    print("predicted token IDs:", predicted_ids)
    print("predicted tokens:", [tokenizer.decode(ids.tolist()) for ids in predicted_ids])


if __name__ == "__main__":
    main()
