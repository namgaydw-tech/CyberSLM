from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class CausalTokenDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Convert token blocks of length N+1 into shifted input/label pairs."""

    def __init__(self, blocks: Iterable[list[int]]) -> None:
        super().__init__()
        self.blocks = blocks

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for block in self.blocks:
            if len(block) < 2:
                continue
            values = torch.tensor(block, dtype=torch.long)
            yield values[:-1], values[1:]
