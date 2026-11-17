from __future__ import annotations

import argparse
import json

from cyberslm.data.loader import ParquetCorpus
from cyberslm.data.schema import DatasetSchema


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a Parquet corpus")
    parser.add_argument("paths", nargs="+", help="Parquet files, directories, or globs")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--document-id-column", default="document_id")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fail-on-quality-issues", action="store_true")
    args = parser.parse_args()
    schema = DatasetSchema(
        text_column=args.text_column,
        document_id_column=args.document_id_column,
    )
    report = ParquetCorpus(args.paths, schema).profile(batch_size=args.batch_size)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.fail_on_quality_issues and (
        report.missing_required_columns
        or report.invalid_utf8_count
        or report.null_counts.get(args.text_column, 0)
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
