import torch

from sparklm.text_utils import text_to_token_ids, token_ids_to_text
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def test_text_token_id_round_trip_preserves_known_text():
    tokenizer = BPETokenizer("gpt2")
    text = "Every effort moves you"

    token_ids = text_to_token_ids(text, tokenizer)

    assert token_ids.shape == (1, len(tokenizer.encode(text)))
    assert token_ids_to_text(token_ids, tokenizer) == text


def test_text_to_token_ids_encodes_endoftext_as_special_token():
    tokenizer = BPETokenizer("gpt2")

    token_ids = text_to_token_ids("<|endoftext|>", tokenizer)

    assert token_ids.shape == (1, 1)
    assert token_ids.dtype == torch.long
    assert token_ids.item() == 50256
