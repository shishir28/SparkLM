from sparklm.tokenizers.basic_tokenizer import BasicTokenizer


def test_tokenize_simple_words():
    pattern = r"\s+"
    tk = BasicTokenizer(pattern)
    text = "Hello, world!"
    tokens = tk.tokenize(text)
    assert tokens == ["Hello,", "world!"], f"unexpected tokens: {tokens}"


def test_encode_decode_round_trip():
    tk = BasicTokenizer(r"\s+")
    text = "apple banana apple"
    ids = tk.encode(text)
    decoded = tk.decode(ids)
    assert ids == [0, 1, 0]
    assert decoded == "apple banana apple"


def test_set_pattern_later():
    tk = BasicTokenizer(r"\s+")
    tk.set_pattern(r"\s+")
    assert tk.tokenize("abc 123") == ["abc", "123"]
