from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Evidence:
    document_id: str
    text: str
    source_url: str
    publisher: str | None = None
    publication_year: int | None = None
    score: float | None = None


class EvidenceRetriever(Protocol):
    """Future retrieval boundary; implementations must return attributable evidence."""

    def retrieve(
        self, query: str, *, limit: int = 5, as_of_year: int | None = None
    ) -> Sequence[Evidence]: ...
