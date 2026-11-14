from pathlib import Path

from cyberslm.data.loader import ParquetCorpus
from cyberslm.data.splits import assign_chronological_splits, assign_document_splits

FIXTURE = Path("data/fixtures/synthetic_cybersecurity.parquet")


def test_fixture_schema_and_profile() -> None:
    corpus = ParquetCorpus(FIXTURE)
    report = corpus.profile(batch_size=2)
    assert report.row_count == 6
    assert report.unique_document_count == 5
    assert not report.missing_required_columns
    assert report.invalid_utf8_count == 0
    assert report.duplicate_text_count == 0
    assert all("Fictional" in text or "Synthetic" in text for text in corpus.iter_text())


def test_grouped_split_is_reproducible_and_has_no_document_leakage() -> None:
    ids = ["doc-a", "doc-a", "doc-b", "doc-c", "doc-d"]
    first = assign_document_splits(
        ids, seed=12, train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25
    )
    second = assign_document_splits(
        reversed(ids), seed=12, train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25
    )
    assert first == second
    assert len(first) == len(set(ids))


def test_chronological_split() -> None:
    result = assign_chronological_splits(
        [("old", 2022), ("validation", 2024), ("test", 2025)],
        validation_year=2024,
        test_year=2025,
    )
    assert [item.split for item in result] == ["train", "test", "validation"]
