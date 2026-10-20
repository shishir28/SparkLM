import pytest
import torch

from sparklm.dataset import GPTDatasetV1


class DummyTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def encode(self, text: str):
        return self.token_ids


def make_dataset(token_count: int, max_length: int = 3, stride: int = 1):
    tokenizer = DummyTokenizer(list(range(token_count)))
    return GPTDatasetV1("unused", tokenizer, max_length=max_length, stride=stride)


def test_gpt_dataset_windows_have_full_length_and_one_token_shift():
    dataset = make_dataset(token_count=6, max_length=3, stride=1)

    assert len(dataset) == 3
    for input_ids, target_ids in dataset:
        assert input_ids.shape == (3,)
        assert target_ids.shape == (3,)
        assert torch.equal(target_ids[:-1], input_ids[1:])


def test_gpt_dataset_uses_expected_start_offsets():
    dataset = make_dataset(token_count=6, max_length=3, stride=2)

    assert len(dataset) == 2
    assert dataset[0][0].tolist() == [0, 1, 2]
    assert dataset[0][1].tolist() == [1, 2, 3]
    assert dataset[1][0].tolist() == [2, 3, 4]
    assert dataset[1][1].tolist() == [3, 4, 5]


def test_exactly_max_length_plus_one_tokens_yields_one_window():
    dataset = make_dataset(token_count=4, max_length=3)

    assert len(dataset) == 1
    assert dataset[0][0].tolist() == [0, 1, 2]
    assert dataset[0][1].tolist() == [1, 2, 3]


def test_exactly_max_length_tokens_yields_no_windows():
    dataset = make_dataset(token_count=3, max_length=3)

    assert len(dataset) == 0


def test_gpt_dataset_invalid_window_raises():
    tokenizer = DummyTokenizer(list(range(6)))

    with pytest.raises(ValueError, match="max_length must be positive"):
        GPTDatasetV1("unused", tokenizer, max_length=0, stride=1)

    with pytest.raises(ValueError, match="stride must be positive"):
        GPTDatasetV1("unused", tokenizer, max_length=3, stride=0)


def test_gpt_dataset_returns_long_tensors():
    input_ids, target_ids = make_dataset(token_count=4)[0]

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long
