from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from cyberslm.tokenization.byte import ByteTokenizer


@dataclass(frozen=True)
class EvidenceExample:
    prompt: str
    answer: str


class AnswerOnlyDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create shifted causal-LM examples with prompt targets masked to -100."""

    def __init__(
        self,
        examples: Sequence[EvidenceExample],
        *,
        tokenizer: ByteTokenizer | None = None,
        max_length: int = 512,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer or ByteTokenizer()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        example = self.examples[index]
        prompt_ids = self.tokenizer.encode(example.prompt)
        answer_ids = self.tokenizer.encode(example.answer)
        full = [self.tokenizer.bos_id, *prompt_ids, *answer_ids, self.tokenizer.eos_id]
        if len(full) > self.max_length:
            raise ValueError(
                f"Example {index} has {len(full)} tokens, exceeding max_length={self.max_length}"
            )
        inputs = torch.tensor(full[:-1], dtype=torch.long)
        labels = torch.tensor(full[1:], dtype=torch.long)
        labels[: len(prompt_ids)] = -100
        return inputs, labels


def pad_answer_only_batch(
    batch: Sequence[tuple[Tensor, Tensor]], *, pad_id: int = 0
) -> tuple[Tensor, Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    maximum = max(inputs.numel() for inputs, _ in batch)
    inputs = torch.full((len(batch), maximum), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maximum), -100, dtype=torch.long)
    for row, (item_inputs, item_labels) in enumerate(batch):
        inputs[row, : item_inputs.numel()] = item_inputs
        labels[row, : item_labels.numel()] = item_labels
    return inputs, labels
