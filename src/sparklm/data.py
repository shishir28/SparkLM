import torch


def make_example(ids: list[int], start: int, block_size: int) -> tuple[list[int], list[int]]:
    if start < 0:
        raise ValueError(f"Start index must be non-negative, got {start}")
    if block_size <= 0:
        raise ValueError(f"Block size must be positive, got {block_size}")

    if start + block_size + 1 > len(ids):
        raise ValueError("Not enough tokens to create input and target")

    x = ids[start : start + block_size]
    y = ids[start + 1 : start + block_size + 1]
    return x, y


def make_batch(ids: list[int], starts: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_batch = []
    y_batch = []
    for start in starts:
        x, y = make_example(ids, start, block_size)
        x_batch.append(x) 
        y_batch.append(y)
    return torch.tensor(x_batch, dtype=torch.long), torch.tensor(y_batch, dtype=torch.long)