from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from cyberslm.data.schema import DatasetSchema


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    missing_supported_columns: tuple[str, ...]
    unsupported_columns: tuple[str, ...]
    null_counts: dict[str, int]
    empty_text_count: int
    duplicate_text_count: int
    invalid_utf8_count: int
    unique_document_count: int
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParquetCorpus:
    """Lazy Arrow dataset facade for one or many Parquet files."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        schema: DatasetSchema | None = None,
    ) -> None:
        self.schema_config = schema or DatasetSchema()
        inputs = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not inputs:
            raise ValueError("At least one Parquet path is required")
        resolved = self._resolve_paths(inputs)
        if not resolved:
            raise FileNotFoundError(f"No Parquet files matched: {inputs}")
        self.paths = tuple(resolved)
        self.dataset = ds.dataset(  # type: ignore[no-untyped-call]
            [str(path) for path in self.paths], format="parquet"
        )

    @staticmethod
    def _resolve_paths(inputs: Sequence[str | Path]) -> list[Path]:
        output: list[Path] = []
        for item in inputs:
            path = Path(item)
            if any(char in str(path) for char in "*?["):
                output.extend(sorted(path.parent.glob(path.name)))
            elif path.is_dir():
                output.extend(sorted(path.rglob("*.parquet")))
            elif path.exists():
                output.append(path)
        return list(dict.fromkeys(p.resolve() for p in output))

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.dataset.schema

    def validate_schema(self) -> tuple[str, ...]:
        names = set(self.arrow_schema.names)
        return tuple(
            column for column in self.schema_config.required_columns if column not in names
        )

    def iter_batches(
        self,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 4096,
        filter_expression: Any | None = None,
    ) -> Iterator[pa.RecordBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        yield from self.dataset.to_batches(
            columns=list(columns) if columns else None,
            batch_size=batch_size,
            filter=filter_expression,
        )

    def iter_text(self, *, batch_size: int = 4096) -> Iterator[str]:
        text_column = self.schema_config.text_column
        if text_column not in self.arrow_schema.names:
            raise ValueError(f"Missing text column: {text_column}")
        for batch in self.iter_batches(columns=[text_column], batch_size=batch_size):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value

    def profile(self, *, batch_size: int = 4096) -> DataQualityReport:
        columns = tuple(self.arrow_schema.names)
        missing_required = self.validate_schema()
        supported = self.schema_config.supported_columns
        missing_supported = tuple(sorted(self.schema_config.expected_columns - set(columns)))
        unsupported = tuple(sorted(set(columns) - supported))
        null_counts: Counter[str] = Counter()
        empty_text_count = 0
        invalid_utf8_count = 0
        text_hashes: Counter[str] = Counter()
        document_ids: set[str] = set()
        row_count = 0
        text_col = self.schema_config.text_column
        doc_col = self.schema_config.document_id_column

        for batch in self.iter_batches(batch_size=batch_size):
            row_count += batch.num_rows
            for name in columns:
                null_counts[name] += batch.column(name).null_count
            if text_col in columns:
                for value in batch.column(text_col).to_pylist():
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        invalid_utf8_count += 1
                        continue
                    try:
                        encoded = value.encode("utf-8", errors="strict")
                    except UnicodeError:
                        invalid_utf8_count += 1
                        continue
                    if not value.strip():
                        empty_text_count += 1
                    else:
                        text_hashes[hashlib.sha256(encoded).hexdigest()] += 1
            if doc_col in columns:
                document_ids.update(
                    str(v) for v in batch.column(doc_col).to_pylist() if v is not None
                )

        duplicates = sum(count - 1 for count in text_hashes.values() if count > 1)
        return DataQualityReport(
            row_count=row_count,
            columns=columns,
            missing_required_columns=missing_required,
            missing_supported_columns=missing_supported,
            unsupported_columns=unsupported,
            null_counts=dict(null_counts),
            empty_text_count=empty_text_count,
            duplicate_text_count=duplicates,
            invalid_utf8_count=invalid_utf8_count,
            unique_document_count=len(document_ids),
            schema=str(self.arrow_schema),
        )

    def count_rows(self) -> int:
        return int(self.dataset.count_rows())
