from __future__ import annotations

import torch


def deterministic_batch_indices(
    *, dataset_size: int, batch_size: int, sample_cursor: int, seed: int
) -> tuple[list[int], int]:
    """Return shuffled indices whose sequence is reproducible from a saved cursor."""
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sample_cursor < 0:
        raise ValueError("sample_cursor cannot be negative")
    selected: list[int] = []
    cursor = sample_cursor
    while len(selected) < batch_size:
        epoch, offset = divmod(cursor, dataset_size)
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(dataset_size, generator=generator).tolist()
        available = min(batch_size - len(selected), dataset_size - offset)
        selected.extend(permutation[offset : offset + available])
        cursor += available
    return selected, cursor
