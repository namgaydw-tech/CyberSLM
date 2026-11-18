from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pypdfium2 as pdfium


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CyberSLM-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    if not content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not PDF: {url}")
    path.write_bytes(content)


def extract_pdf(path: Path) -> tuple[str, int]:
    pages: list[str] = []
    document = pdfium.PdfDocument(path)
    try:
        for index in range(len(document)):
            page = document[index]
            text_page = page.get_textpage()
            text = normalize_text(text_page.get_text_range() or "")
            if text:
                pages.append(text)
            text_page.close()
            page.close()
        return "\n\n".join(pages), len(document)
    finally:
        document.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build licensed NIST Stage A corpus")
    parser.add_argument("--sources", type=Path, default=Path("configs/stage_a_sources.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = json.loads(args.sources.read_text(encoding="utf-8"))
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_text_hashes: set[str] = set()
    for source in config["sources"]:
        pdf_path = raw_dir / f"{source['source_id']}.pdf"
        if args.force_download or not pdf_path.exists():
            print(f"Downloading {source['source_id']} ...", flush=True)
            download(source["url"], pdf_path)
        text, pages = extract_pdf(pdf_path)
        if len(text.encode("utf-8")) < 10_000:
            raise ValueError(f"Extraction unexpectedly small for {source['source_id']}")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            raise ValueError(f"Duplicate extracted document: {source['source_id']}")
        seen_text_hashes.add(text_hash)
        rows.append(
            {
                "document_id": source["source_id"],
                "split": source["split"],
                "title": source["title"],
                "source_url": source["url"],
                "retrieval_date": config["retrieval_date"],
                "license_status": config["license_evidence"]["status"],
                "pdf_sha256": sha256_file(pdf_path),
                "text_sha256": text_hash,
                "pages": pages,
                "utf8_bytes": len(text.encode("utf-8")),
                "preprocessing": (
                    "PDFium page extraction; soft-hyphen/NUL removal; whitespace normalization"
                ),
                "text": text,
            }
        )
        print(f"Extracted {source['source_id']}: {pages} pages, {len(text):,} chars")

    frame = pd.DataFrame(rows).sort_values("document_id").reset_index(drop=True)
    if set(frame.groupby("document_id")["split"].nunique()) != {1}:
        raise ValueError("document_id crosses corpus splits")
    corpus_path = args.output_dir / "corpus.parquet"
    frame.to_parquet(corpus_path, index=False)
    manifest = {
        "corpus_id": config["corpus_id"],
        "retrieval_date": config["retrieval_date"],
        "license_evidence": config["license_evidence"],
        "source_config_sha256": sha256_file(args.sources),
        "corpus_parquet_sha256": sha256_file(corpus_path),
        "documents": len(frame),
        "split_documents": frame["split"].value_counts().sort_index().to_dict(),
        "split_utf8_bytes": frame.groupby("split")["utf8_bytes"].sum().to_dict(),
        "total_utf8_bytes": int(frame["utf8_bytes"].sum()),
        "records": frame.drop(columns=["text"]).to_dict("records"),
        "deduplication": "exact SHA-256 of normalized extracted document text",
        "split_rule": "frozen source-document split from configs/stage_a_sources.json",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
