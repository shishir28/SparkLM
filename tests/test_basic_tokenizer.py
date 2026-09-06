from sparklm.tokenizers.basic_tokenizer import BasicTokenizer


def test_tokenize_simple_words():
    # Split on whitespace; punctuation stays attached to the neighboring word.
    pattern = r"\s+"
    tk = BasicTokenizer(pattern)
    text = "Hello, world!"
    tokens = tk.tokenize(text)
    assert tokens == ["Hello,", "world!"], f"unexpected tokens: {tokens}"


def test_set_pattern_later():
    tk = BasicTokenizer()
    try:
        tk.tokenize("no pattern")
        assert False, "tokenize should have raised when pattern is not set"
    except ValueError:
        pass
    tk.set_pattern(r"\s+")
    assert tk.tokenize("abc 123") == ["abc", "123"]
