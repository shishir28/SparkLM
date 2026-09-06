from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from sparklm.dataset import GPTDatasetV1
from sparklm.tokenizers.bpe_tokenizer import BPETokenizer


def create_dataloader_v1(
    text: str,
    tokenizer: Optional[BPETokenizer] = None,
    batch_size: int = 4, #number of examples packed into one DataLoader batch
    max_length: int = 256, #length of training window
    stride: int = 128, # how far the window moves, smaller stride means more overlapping windows, bigger stride means less overlapping windows
    shuffle: bool = True, # changes the order of windows 
    drop_last: bool = True, # Final short batch is discarded if True, otherwise it is returned as a smaller batch
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the book's sliding-window next-token dataset."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    if tokenizer is None:
        tokenizer = BPETokenizer("gpt2")

    dataset = GPTDatasetV1(text=text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
