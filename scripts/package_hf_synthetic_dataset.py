from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_template(path: Path, values: dict[str, str]) -> str:
    content = path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace("{{" + key + "}}", value)
    if "{{" in content or "}}" in content:
        raise ValueError("Unresolved dataset-card template placeholder")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Package publishable synthetic HF dataset")
    parser.add_argument(
        "--input", type=Path, default=Path("data/fixtures/synthetic_cybersecurity.parquet")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="namgaydw-ai/CyberSLM-Evidence-QA")
    parser.add_argument("--template", type=Path, default=Path("docs/templates/hf_dataset_card.md"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    required = {
        "document_id",
        "chunk_id",
        "text",
        "source_url",
        "publisher",
        "report_year",
        "license",
    }
    if required - set(frame.columns):
        raise ValueError(f"Missing fixture fields: {sorted(required - set(frame.columns))}")
    if not frame["license"].str.contains("Synthetic test fixture", case=False).all():
        raise ValueError("Every row must be explicitly marked as a synthetic fixture")
    if not frame["source_url"].str.contains(r"\.invalid/", regex=True).all():
        raise ValueError("Every source URL must use the reserved .invalid domain")
    if frame["chunk_id"].duplicated().any():
        raise ValueError("chunk_id values must be unique")

    documents = sorted(frame["document_id"].unique())
    assignments = {
        document: (
            "test"
            if index == len(documents) - 1
            else "validation"
            if index == len(documents) - 2
            else "train"
        )
        for index, document in enumerate(documents)
    }
    frame = frame.copy()
    frame["split"] = frame["document_id"].map(assignments)
    if (frame.groupby("document_id")["split"].nunique() != 1).any():
        raise ValueError("A document crosses dataset splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        subset = frame[frame["split"] == split].reset_index(drop=True)
        subset.to_parquet(args.output_dir / f"{split}.parquet", index=False)
        split_counts[split] = len(subset)
    manifest = {
        "dataset_id": "cyberslm_synthetic_fixture_v1",
        "repo_id": args.repo_id,
        "status": "unmistakably fictional public demonstration fixture",
        "source_fixture_sha256": sha256_file(args.input),
        "records": len(frame),
        "split_counts": split_counts,
        "document_split": assignments,
        "checks": {
            "all_rows_marked_synthetic": True,
            "all_urls_reserved_invalid": True,
            "unique_chunk_ids": True,
            "document_group_independence": True,
        },
        "not_the_phase4f_dataset": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card = render_template(
        args.template,
        {
            "TOTAL_RECORDS": str(len(frame)),
            "TRAIN_RECORDS": str(split_counts["train"]),
            "VALIDATION_RECORDS": str(split_counts["validation"]),
            "TEST_RECORDS": str(split_counts["test"]),
            "MANIFEST_SHA256": sha256_file(manifest_path),
            "REPO_ID": args.repo_id,
        },
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    receipt_files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "checksums.json"
    )
    receipts = {path.name: sha256_file(path) for path in receipt_files}
    (args.output_dir / "checksums.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
