"""Explore greedy, temperature, top-k, and EOS decoding."""

from sparklm.config import GPT_CONFIG_124M
from sparklm.generation import generate, generate_text_simple, softmax_with_temperature
from sparklm.model import GPTModel
from sparklm.text_utils import token_ids_to_text
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer

import torch

    # Create a toy vocabulary with 9 token IDs.
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

def main() -> int:
    # Import PyTorch and the decoding functions you want to demonstrate.

    # Define fixed logits or probabilities for the toy vocabulary.
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)


    # Select one token with argmax and print the result.
    next_token_id = torch.argmax(probas).item()
    print(f"Greedy selection: {next_token_id} -> {inverse_vocab[next_token_id]}")

    # Sample from the distribution 1,000 times with torch.multinomial.
    print_sampled_tokens(probas)

    # Apply temperatures 1, 0.1, and 5 and print the distributions.

    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

    print(scaled_probas)

    # Apply top-k filtering with k=3.

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    # Print the filtered logits and verify that only the top-k remain finite.
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)

    # Demonstrate greedy generation with the existing reference generator.
    tokenizer = BPETokenizer()
    model = GPTModel(GPT_CONFIG_124M).eval()
    token_ids = torch.tensor([tokenizer.encode("closer every effort forward inches moves pizza toward you")], dtype=torch.long)

    generated_ids = generate_text_simple(
          model,
          token_ids,
          max_new_tokens=6,
          context_size= GPT_CONFIG_124M["context_length"],
    )
    print("Generated token IDs:", generated_ids)
    # Demonstrate temperature-based sampling with the general generator.
    generated_token_ids_1 = generate(model,
             token_ids,
             max_new_tokens=6,
             context_size=GPT_CONFIG_124M["context_length"],
             temperature=0.0,
             top_k=None,
             eos_id=None)
    print("Generated text with temperature=0.0:", token_ids_to_text(generated_token_ids_1,tokenizer))

    # Demonstrate top-k sampling with the general generator.
    generated_token_ids_2 = generate(model,
             token_ids,
             max_new_tokens=6,
             context_size=GPT_CONFIG_124M["context_length"],
             temperature=1.0,
             top_k=3,
             eos_id=None)
    print("Generated text with top_k=3:", token_ids_to_text(generated_token_ids_2,tokenizer))
    # Demonstrate early stopping with an EOS token.
    generated_token_ids_3 = generate(model,
             token_ids,
             max_new_tokens=6,
             context_size=GPT_CONFIG_124M["context_length"],
             temperature=1.0,
             top_k=None,
             eos_id=tokenizer.encode(".")[0])
    print("Generated text with EOS token:", token_ids_to_text(generated_token_ids_3,tokenizer))
    return 0


def print_sampled_tokens(probas):
    torch.manual_seed(123)  # For reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")



if __name__ == "__main__":
    raise SystemExit(main())
