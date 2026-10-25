import torch

from sparklm.config import GPT2_MODEL_CONFIGS
from sparklm.generation import generate_text_simple
from sparklm.gpt2_weights import load_weights_into_gpt
from sparklm.model import GPTModel
from sparklm.text_utils import text_to_token_ids, token_ids_to_text


CHOOSE_MODEL = "gpt2-small (124M)"
CLASSIFICATION_PROMPT = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    **GPT2_MODEL_CONFIGS[CHOOSE_MODEL],
}


def initialize_pretrained_gpt2(models_dir):
    from sparklm.gpt_download import download_and_load_gpt2

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size, models_dir)

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    return model


def run_pretrained_sanity_checks(model, tokenizer):
    prompts = (
        ("Every effort moves you", 15),
        (CLASSIFICATION_PROMPT, 23),
    )
    generated_texts = []
    for prompt, max_new_tokens in prompts:
        token_ids = generate_text_simple(
            model,
            idx=text_to_token_ids(prompt, tokenizer),
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        generated_texts.append(generated_text)
        print(f"Generated text for '{prompt}': {generated_text}")
    return tuple(generated_texts)


def add_classification_head(model, num_classes=2, trainable_last_blocks=1):
    num_blocks = len(model.transformer_blocks)
    if not 1 <= trainable_last_blocks <= num_blocks:
        raise ValueError(
            "trainable_last_blocks must be between 1 and "
            f"{num_blocks}, found {trainable_last_blocks}"
        )

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(
        in_features=model.out_head.in_features,
        out_features=num_classes,
    )

    for block in model.transformer_blocks[-trainable_last_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.final_layer_norm.parameters():
        param.requires_grad = True

    return model
