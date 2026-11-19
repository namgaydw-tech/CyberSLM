from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypdfium2 as pdfium

from cyberslm.rag import ollama_embeddings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str) -> bytes:
    curl = subprocess.run(
        [
            "curl",
            "-fsSL",
            "-A",
            "Mozilla/5.0",
            "-e",
            "https://www.cisa.gov/",
            url,
        ],
        capture_output=True,
        check=False,
    )
    if curl.returncode == 0 and curl.stdout:
        return curl.stdout
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            ),
            "Referer": "https://www.cisa.gov/",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"\[(https?://[^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(content: bytes, path: Path) -> tuple[str, int]:
    path.write_bytes(content)
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


def chunk_text(text: str, *, size: int = 1400, overlap: int = 220) -> list[str]:
    if size <= overlap or overlap < 0:
        raise ValueError("Chunk size must exceed non-negative overlap")
    normalized = normalize_text(text)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            candidates = [
                normalized.rfind(marker, start + size // 2, end) for marker in ("\n", ". ")
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if len(chunk) >= 120:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def nist_documents(source: dict[str, Any], root: Path) -> list[dict[str, str]]:
    path = root / str(source["path"])
    frame = pd.read_parquet(path)
    return [
        {
            "document_id": str(row.document_id),
            "title": str(row.title),
            "source_url": str(row.source_url),
            "license": str(row.license_status),
            "attribution": "National Institute of Standards and Technology",
            "text": str(row.text),
        }
        for row in frame.itertuples(index=False)
    ]


def mitre_documents(source: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    content = download(str(source["url"]))
    bundle: dict[str, Any] = json.loads(content)
    documents: list[dict[str, str]] = [
        {
            "document_id": "mitre_attack_enterprise_overview",
            "title": "MITRE ATT&CK Enterprise Framework Overview",
            "source_url": "https://attack.mitre.org/",
            "license": str(source["license"]),
            "attribution": str(source["attribution"]),
            "text": str(source["overview"]),
        }
    ]
    allowed = {"attack-pattern", "course-of-action", "malware", "tool"}
    for item in bundle.get("objects", []):
        if item.get("type") not in allowed or item.get("revoked") or item.get("x_mitre_deprecated"):
            continue
        description = normalize_text(str(item.get("description", "")))
        if not description:
            continue
        references = item.get("external_references", [])
        attack_ref = next(
            (ref for ref in references if str(ref.get("source_name")) == "mitre-attack"), {}
        )
        external_id = str(attack_ref.get("external_id", item.get("id", "unknown")))
        source_url = str(
            attack_ref.get("url", f"https://attack.mitre.org/techniques/{external_id}/")
        )
        name = str(item.get("name", external_id))
        documents.append(
            {
                "document_id": f"mitre_{external_id.lower().replace('.', '_')}",
                "title": f"MITRE ATT&CK {external_id}: {name}",
                "source_url": source_url,
                "license": str(source["license"]),
                "attribution": str(source["attribution"]),
                "text": f"{name} ({external_id})\n\n{description}",
            }
        )
    return documents, hashlib.sha256(content).hexdigest()


def pdf_document(source: dict[str, Any], raw_dir: Path) -> tuple[dict[str, str], str, int]:
    content = download(str(source["url"]))
    if not content.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF from {source['url']}")
    pdf_path = raw_dir / f"{source['source_id']}.pdf"
    text, pages = extract_pdf(content, pdf_path)
    document = {
        "document_id": str(source["source_id"]),
        "title": str(source["title"]),
        "source_url": str(source["url"]),
        "license": str(source["license"]),
        "attribution": "Cybersecurity and Infrastructure Security Agency (CISA)",
        "text": text,
    }
    return document, hashlib.sha256(content).hexdigest(), pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CyberSLM trusted-knowledge embeddings")
    parser.add_argument("--config", type=Path, default=Path("configs/rag_sources.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="all-minilm:latest")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = Path.cwd()
    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    documents: list[dict[str, str]] = []
    source_receipts: list[dict[str, Any]] = []
    for source in config["sources"]:
        kind = source["kind"]
        if kind == "local_parquet":
            loaded = nist_documents(source, root)
            documents.extend(loaded)
            source_receipts.append(
                {
                    "source_id": source["source_id"],
                    "kind": kind,
                    "documents": len(loaded),
                    "sha256": sha256_file(root / source["path"]),
                    "license": source["license"],
                }
            )
        elif kind == "mitre_attack_stix":
            loaded, digest = mitre_documents(source)
            documents.extend(loaded)
            source_receipts.append(
                {
                    "source_id": source["source_id"],
                    "kind": kind,
                    "documents": len(loaded),
                    "sha256": digest,
                    "license": source["license"],
                    "attribution": source["attribution"],
                }
            )
        elif kind == "pdf":
            loaded, digest, pages = pdf_document(source, raw_dir)
            documents.append(loaded)
            source_receipts.append(
                {
                    "source_id": source["source_id"],
                    "kind": kind,
                    "documents": 1,
                    "pages": pages,
                    "sha256": digest,
                    "license": source["license"],
                }
            )
        else:
            raise ValueError(f"Unsupported source kind: {kind}")
        print(f"Loaded {source['source_id']}", flush=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(
                {
                    "chunk_id": f"{document['document_id']}:{index:05d}",
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source_url": document["source_url"],
                    "license": document["license"],
                    "attribution": document["attribution"],
                    "text_sha256": digest,
                    "text": text,
                }
            )
    frame = pd.DataFrame(rows)
    vectors: list[np.ndarray] = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]["text"].tolist()
        vectors.append(ollama_embeddings(batch, model=args.embedding_model))
        print(f"Embedded {min(start + len(batch), len(frame))}/{len(frame)}", flush=True)
    embeddings = np.concatenate(vectors, axis=0).astype(np.float32)
    chunks_path = args.output_dir / "chunks.parquet"
    embeddings_path = args.output_dir / "embeddings.npy"
    frame.to_parquet(chunks_path, index=False)
    np.save(embeddings_path, embeddings)
    manifest = {
        "status": "PASS",
        "collection_id": config["collection_id"],
        "retrieval_date": config["retrieval_date"],
        "documents": len(documents),
        "chunks": len(frame),
        "embedding_model": args.embedding_model,
        "embedding_dimensions": int(embeddings.shape[1]),
        "chunks_sha256": sha256_file(chunks_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "source_config_sha256": sha256_file(args.config),
        "sources": source_receipts,
        "privacy": "Public technical sources only; no private reports or event records",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
