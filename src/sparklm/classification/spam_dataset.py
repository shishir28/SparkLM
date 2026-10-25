from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        max_length=None,
        pad_token_id=50256,
    ):
        self.data = pd.read_csv(csv_file)
        if self.data.empty:
            raise ValueError("SpamDataset requires at least one row")

        encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = max(len(text) for text in encoded_texts)
        elif not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        else:
            self.max_length = max_length

        self.encoded_texts = []
        for encoded_text in encoded_texts:
            encoded_text = encoded_text[: self.max_length]
            padding_length = self.max_length - len(encoded_text)
            self.encoded_texts.append(
                encoded_text + [pad_token_id] * padding_length
            )

    def __getitem__(self, index):
        encoded_text = torch.tensor(
            self.encoded_texts[index],
            dtype=torch.long,
        )
        label = torch.tensor(
            self.data["Label"].iloc[index],
            dtype=torch.long,
        )
        return encoded_text, label

    def __len__(self):
        return len(self.data)


def create_spam_datasets(
    train_csv,
    validation_csv,
    test_csv,
    tokenizer,
    context_length=1024,
):
    paths = [Path(train_csv), Path(validation_csv), Path(test_csv)]
    train_dataset = SpamDataset(paths[0], tokenizer)
    if train_dataset.max_length > context_length:
        raise ValueError(
            f"Training max_length {train_dataset.max_length} exceeds "
            f"context length {context_length}"
        )

    validation_dataset = SpamDataset(
        paths[1],
        tokenizer,
        max_length=train_dataset.max_length,
    )
    test_dataset = SpamDataset(
        paths[2],
        tokenizer,
        max_length=train_dataset.max_length,
    )
    return train_dataset, validation_dataset, test_dataset


def create_spam_data_loaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=8,
    num_workers=0,
    seed=123,
):
    torch.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader, test_loader
