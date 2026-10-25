import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sparklm.classification.spam_dataset import (
    SpamDataset,
    create_spam_data_loaders,
    create_spam_datasets,
)


class CharacterTokenizer:
    def __init__(self):
        self.encoded_inputs = []

    def encode(self, text):
        self.encoded_inputs.append(text)
        return [ord(character) for character in text]


@pytest.fixture
def spam_csv(tmp_path):
    path = tmp_path / "spam.csv"
    pd.DataFrame(
        {
            "Text": ["a", "abc", "abcdef"],
            "Label": [0, 1, 0],
        }
    ).to_csv(path, index=False)
    return path


def test_infers_longest_length_and_encodes_each_text_once(spam_csv):
    tokenizer = CharacterTokenizer()

    dataset = SpamDataset(spam_csv, tokenizer)

    assert dataset.max_length == 6
    assert tokenizer.encoded_inputs == ["a", "abc", "abcdef"]
    assert all(len(encoded) == 6 for encoded in dataset.encoded_texts)


def test_right_padding_preserves_encoded_prefix(spam_csv):
    dataset = SpamDataset(spam_csv, CharacterTokenizer())

    assert dataset.encoded_texts[0] == [ord("a")] + [50256] * 5
    assert dataset.encoded_texts[1] == [
        ord("a"),
        ord("b"),
        ord("c"),
        50256,
        50256,
        50256,
    ]


def test_explicit_max_length_truncates_and_pads(spam_csv):
    dataset = SpamDataset(
        spam_csv,
        CharacterTokenizer(),
        max_length=4,
    )

    assert dataset.max_length == 4
    assert dataset.encoded_texts[0] == [ord("a"), 50256, 50256, 50256]
    assert dataset.encoded_texts[2] == [
        ord("a"),
        ord("b"),
        ord("c"),
        ord("d"),
    ]


def test_items_are_long_tensors_with_scalar_labels(spam_csv):
    dataset = SpamDataset(spam_csv, CharacterTokenizer())

    encoded_text, label = dataset[1]

    assert encoded_text.shape == (6,)
    assert encoded_text.dtype == torch.long
    assert label.shape == ()
    assert label.dtype == torch.long
    assert label.item() == 1


def test_validation_dataset_reuses_training_max_length(tmp_path, spam_csv):
    validation_path = tmp_path / "validation.csv"
    pd.DataFrame(
        {"Text": ["abcdefgh", "xy"], "Label": [1, 0]}
    ).to_csv(validation_path, index=False)
    train_dataset = SpamDataset(spam_csv, CharacterTokenizer())

    validation_dataset = SpamDataset(
        validation_path,
        CharacterTokenizer(),
        max_length=train_dataset.max_length,
    )

    assert validation_dataset.max_length == train_dataset.max_length
    assert validation_dataset.encoded_texts[0] == [
        ord(character) for character in "abcdef"
    ]
    assert len(validation_dataset.encoded_texts[1]) == train_dataset.max_length


def test_data_loader_produces_batch_by_length_inputs(spam_csv):
    dataset = SpamDataset(spam_csv, CharacterTokenizer())
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    inputs, labels = next(iter(loader))

    assert inputs.shape == (2, 6)
    assert labels.shape == (2,)
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long


def test_rejects_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["Text", "Label"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least one row"):
        SpamDataset(path, CharacterTokenizer())


@pytest.mark.parametrize("max_length", [0, -1, 1.5, "4"])
def test_rejects_invalid_max_length(spam_csv, max_length):
    with pytest.raises(ValueError, match="positive integer"):
        SpamDataset(
            spam_csv,
            CharacterTokenizer(),
            max_length=max_length,
        )


def test_create_spam_datasets_reuses_training_length(tmp_path):
    paths = []
    for name, texts in (
        ("train", ["a", "abcd"]),
        ("validation", ["abcdefgh"]),
        ("test", ["xy"]),
    ):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"Text": texts, "Label": [0] * len(texts)}).to_csv(
            path,
            index=False,
        )
        paths.append(path)

    datasets = create_spam_datasets(
        *paths,
        tokenizer=CharacterTokenizer(),
        context_length=10,
    )

    assert [dataset.max_length for dataset in datasets] == [4, 4, 4]
    assert [len(dataset.encoded_texts[0]) for dataset in datasets] == [4, 4, 4]


def test_create_spam_datasets_rejects_training_length_over_context(
    tmp_path,
):
    paths = []
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"Text": ["abcdef"], "Label": [0]}).to_csv(
            path,
            index=False,
        )
        paths.append(path)

    with pytest.raises(ValueError, match="exceeds context length"):
        create_spam_datasets(
            *paths,
            tokenizer=CharacterTokenizer(),
            context_length=5,
        )


def test_create_spam_data_loaders_uses_chapter_six_settings(tmp_path):
    datasets = []
    for name, size in (("train", 10), ("validation", 5), ("test", 6)):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(
            {
                "Text": [f"message-{index}" for index in range(size)],
                "Label": [index % 2 for index in range(size)],
            }
        ).to_csv(path, index=False)
        datasets.append(
            SpamDataset(path, CharacterTokenizer(), max_length=12)
        )

    train_loader, validation_loader, test_loader = create_spam_data_loaders(
        *datasets,
        batch_size=4,
        num_workers=0,
        seed=123,
    )

    assert [len(loader) for loader in (
        train_loader,
        validation_loader,
        test_loader,
    )] == [2, 2, 2]
    assert train_loader.drop_last is True
    assert validation_loader.drop_last is False
    assert test_loader.drop_last is False
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    inputs, labels = next(iter(train_loader))
    assert inputs.shape == (4, 12)
    assert labels.shape == (4,)
