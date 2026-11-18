from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from cyberslm.tokenization.byte import ByteTokenizer


class PackedDocumentDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pack document-separated byte streams into fixed causal-LM blocks."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        tokenizer: ByteTokenizer | None = None,
        sequence_length: int = 512,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer or ByteTokenizer()
        stream: list[int] = []
        for text in documents:
            stream.extend(self.tokenizer.encode(text, bos=True, eos=True))
        self.unique_token_count = len(stream)
        block_width = sequence_length + 1
        usable = (len(stream) // block_width) * block_width
        if usable == 0:
            raise ValueError("Corpus is too small for one packed block")
        self.blocks = torch.tensor(stream[:usable], dtype=torch.long).view(-1, block_width)

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        block = self.blocks[index]
        return block[:-1], block[1:]
