
import pytest
from sparklm.tokenizers.char_tokenizer import CharacterTokenizer


def test_round_trip():
    text = "banana"
    tokenizer = CharacterTokenizer(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_vocab_size_count_unique_characters():
    text = "banana"
    tokenizer = CharacterTokenizer(text)
    assert tokenizer.vocab_size == 3

def test_encode_unknown_character_raises_value_error():
    text = "banana"
    tokenizer = CharacterTokenizer(text)
    with pytest.raises(ValueError):
        tokenizer.encode("apple")

def test_decode_unknown_id_raises_value_error():
    text = "banana"
    tokenizer = CharacterTokenizer(text)
    with pytest.raises(ValueError):
        tokenizer.decode([5])    