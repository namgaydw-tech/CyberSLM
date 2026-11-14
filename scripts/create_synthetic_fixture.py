"""Create a deterministic, fictional Parquet fixture for tests only."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    output = Path(__file__).parents[1] / "data" / "fixtures" / "synthetic_cybersecurity.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "document_id": "fictional-report-alpha",
            "chunk_id": "alpha-001",
            "text": (
                "Fictional lab note: CVE-2099-0001 affects demo.example.invalid on TCP port 443."
            ),
            "source_url": "https://example.invalid/reports/alpha",
            "publisher": "Fictional Security Lab",
            "report_year": 2099,
            "license": "Synthetic test fixture; no real-world claims",
        },
        {
            "document_id": "fictional-report-alpha",
            "chunk_id": "alpha-002",
            "text": "".join(
                (
                    "Synthetic log: 192.0.2.10 denied login for demo-user; ",
                    "action=block severity=low.",
                )
            ),
            "source_url": "https://example.invalid/reports/alpha",
            "publisher": "Fictional Security Lab",
            "report_year": 2099,
            "license": "Synthetic test fixture; no real-world claims",
        },
        {
            "document_id": "fictional-report-beta",
            "chunk_id": "beta-001",
            "text": (
                "Fictional guidance: rotate the example API credential and review least privilege."
            ),
            "source_url": "https://example.invalid/reports/beta",
            "publisher": "Imaginary Defensive Research",
            "report_year": 2100,
            "license": "Synthetic test fixture; no real-world claims",
        },
        {
            "document_id": "fictional-report-gamma",
            "chunk_id": "gamma-001",
            "text": (
                "Synthetic DNS event: sample.example.invalid resolved to 198.51.100.20 in a lab."
            ),
            "source_url": "https://example.invalid/reports/gamma",
            "publisher": "Imaginary Defensive Research",
            "report_year": 2101,
            "license": "Synthetic test fixture; no real-world claims",
        },
        {
            "document_id": "fictional-report-delta",
            "chunk_id": "delta-001",
            "text": (
                "Fictional incident exercise: isolate the test host, preserve evidence, "
                "and document actions."
            ),
            "source_url": "https://example.invalid/reports/delta",
            "publisher": "Example Blue Team Academy",
            "report_year": 2102,
            "license": "Synthetic test fixture; no real-world claims",
        },
        {
            "document_id": "fictional-report-epsilon",
            "chunk_id": "epsilon-001",
            "text": "".join(
                (
                    "Synthetic alert: SHA256 deadbeef is deliberately incomplete ",
                    "and not a real indicator.",
                )
            ),
            "source_url": "https://example.invalid/reports/epsilon",
            "publisher": "Example Blue Team Academy",
            "report_year": 2103,
            "license": "Synthetic test fixture; no real-world claims",
        },
    ]
    schema = pa.schema(
        [
            ("document_id", pa.string()),
            ("chunk_id", pa.string()),
            ("text", pa.string()),
            ("source_url", pa.string()),
            ("publisher", pa.string()),
            ("report_year", pa.int32()),
            ("license", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output, compression="zstd")
    print(f"wrote {len(rows)} fictional rows to {output}")


if __name__ == "__main__":
    main()
