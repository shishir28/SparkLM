"""Load the official GPT-2 124M checkpoint and generate a sample."""

import torch

from sparklm.config import GPT2_MODEL_CONFIGS, GPT_CONFIG_124M
from sparklm.generation import generate
from sparklm.gpt2_weights import load_weights_into_gpt
from sparklm.gpt_download import download_and_load_gpt2
from sparklm.model import GPTModel
from sparklm.text_utils import text_to_token_ids, token_ids_to_text
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def main() -> int:
    settings, params = download_and_load_gpt2("124M", "gpt2")
    config = dict(GPT_CONFIG_124M)
    config.update(GPT2_MODEL_CONFIGS["gpt2-small (124M)"])
    config.update({"context_length": settings["n_ctx"], "qkv_bias": True})

    model = GPTModel(config)
    load_weights_into_gpt(model, params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = BPETokenizer("gpt2")
    token_ids = text_to_token_ids("Every effort moves you", tokenizer).to(device)

    torch.manual_seed(123)
    with torch.no_grad():
        generated_ids = generate(
            model,
            token_ids,
            max_new_tokens=25,
            context_size=config["context_length"],
            top_k=50,
            temperature=1.5,
        )

    print(f"device: {device}")
    print(token_ids_to_text(generated_ids.cpu(), tokenizer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
