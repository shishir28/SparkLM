import torch

from sparklm.dataset import GPTDatasetV1


class DummyTokenizer:
    def encode(self, text: str):
        return [0, 1, 2, 3, 4, 5]


def test_gpt_dataset_shapes_and_shift():
    ds = GPTDatasetV1("hello world", DummyTokenizer(), max_length=3, stride=2)

    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (3,)
    assert y.shape == (3,)
    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]


def test_gpt_dataset_invalid_window_raises():
    try:
        GPTDatasetV1("abc", DummyTokenizer(), max_length=0, stride=1)
        assert False, "Expected ValueError for max_length <= 0"
    except ValueError:
        pass

    try:
        GPTDatasetV1("abc", DummyTokenizer(), max_length=3, stride=0)
        assert False, "Expected ValueError for stride <= 0"
    except ValueError:
        pass


def test_gpt_dataset_returns_long_tensors():
    ds = GPTDatasetV1("abcde", DummyTokenizer(), max_length=3, stride=1)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long
