from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    prompt: str
    reference_answer: str | None
    source_urls: tuple[str, ...]
    publication_year: int | None
    category: str


@dataclass(frozen=True)
class BenchmarkScore:
    metric: str
    value: float
    example_count: int


class Benchmark(Protocol):
    """Interface for future QA, grounding, temporal, and factuality benchmarks."""

    name: str

    def examples(self) -> Sequence[BenchmarkExample]: ...

    def score(self, predictions: Sequence[str]) -> Sequence[BenchmarkScore]: ...
