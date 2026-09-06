from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def test_tiktoken_round_trip():
    tokenizer = BPETokenizer("gpt2")
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_tiktoken_vocab_size_positive():
    tokenizer = BPETokenizer("gpt2")
    assert tokenizer.vocab_size() > 0
