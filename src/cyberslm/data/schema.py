from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSchema:
    """Logical source schema. Only text and document ID are mandatory."""

    text_column: str = "text"
    document_id_column: str = "document_id"
    source_url_column: str = "source_url"
    publisher_column: str = "publisher"
    report_year_column: str = "report_year"
    provenance_columns: tuple[str, ...] = field(
        default=("source_url", "publisher", "report_year", "document_id")
    )
    optional_columns: tuple[str, ...] = field(
        default=(
            "chunk_id",
            "published_at",
            "title",
            "license",
            "content_hash",
            "extraction_method",
            "language",
        )
    )

    @property
    def required_columns(self) -> tuple[str, str]:
        return (self.text_column, self.document_id_column)

    @property
    def supported_columns(self) -> set[str]:
        return {
            self.text_column,
            self.document_id_column,
            self.source_url_column,
            self.publisher_column,
            self.report_year_column,
            *self.provenance_columns,
            *self.optional_columns,
        }

    @property
    def expected_columns(self) -> set[str]:
        return {
            self.text_column,
            self.document_id_column,
            self.source_url_column,
            self.publisher_column,
            self.report_year_column,
            *self.provenance_columns,
        }
