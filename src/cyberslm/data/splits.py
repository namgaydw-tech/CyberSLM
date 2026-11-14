from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SplitAssignment:
    document_id: str
    split: SplitName


def _validate_fractions(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) < 0:
        raise ValueError("Split fractions cannot be negative")
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")


def assign_document_splits(
    document_ids: Iterable[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> list[SplitAssignment]:
    """Assign whole documents deterministically with a stable hash."""
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    train_cutoff = train_fraction
    validation_cutoff = train_fraction + validation_fraction
    assignments: list[SplitAssignment] = []
    for document_id in sorted(set(document_ids)):
        digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        split: SplitName
        if value < train_cutoff:
            split = "train"
        elif value < validation_cutoff:
            split = "validation"
        else:
            split = "test"
        assignments.append(SplitAssignment(document_id, split))
    return assignments


def assign_chronological_splits(
    document_years: Iterable[tuple[str, int]],
    *,
    validation_year: int,
    test_year: int,
) -> list[SplitAssignment]:
    """Reserve newer documents for temporal validation and test sets."""
    if test_year < validation_year:
        raise ValueError("test_year must be greater than or equal to validation_year")
    by_document: dict[str, int] = {}
    for document_id, year in document_years:
        if document_id in by_document and by_document[document_id] != year:
            raise ValueError(f"Conflicting years for document {document_id}")
        by_document[document_id] = year
    result: list[SplitAssignment] = []
    for document_id, year in sorted(by_document.items()):
        split: SplitName = (
            "test" if year >= test_year else "validation" if year >= validation_year else "train"
        )
        result.append(SplitAssignment(document_id, split))
    return result
