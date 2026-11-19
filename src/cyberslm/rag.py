from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    text: str
    score: float


def ollama_embeddings(
    texts: list[str],
    *,
    model: str = "all-minilm:latest",
    url: str = "http://127.0.0.1:11434/api/embed",
    timeout: int = 180,
) -> np.ndarray:
    if not texts:
        raise ValueError("At least one text is required for embedding")
    payload = json.dumps({"model": model, "input": texts, "truncate": True}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.load(response)
    vectors = np.asarray(result["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("Embedding service returned an unexpected shape")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return np.asarray(normalized, dtype=np.float32)


class KnowledgeRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedding_model: str = "all-minilm:latest",
        embedding_url: str = "http://127.0.0.1:11434/api/embed",
    ) -> None:
        directory = Path(index_dir)
        self.frame = pd.read_parquet(directory / "chunks.parquet")
        self.embeddings = np.load(directory / "embeddings.npy")
        if len(self.frame) != len(self.embeddings):
            raise ValueError("Chunk metadata and embedding counts differ")
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {
            "about",
            "and",
            "are",
            "common",
            "does",
            "explain",
            "for",
            "from",
            "how",
            "the",
            "this",
            "what",
            "when",
            "why",
            "with",
        }
        return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2} - stop

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        query_vector = ollama_embeddings(
            [query], model=self.embedding_model, url=self.embedding_url
        )[0]
        dense_scores = self.embeddings @ query_vector
        query_terms = self._terms(query)
        lexical = np.zeros(len(self.frame), dtype=np.float32)
        if query_terms:
            for index, row in enumerate(self.frame.itertuples(index=False)):
                text_terms = self._terms(f"{row.title} {row.text}")
                title_terms = self._terms(str(row.title))
                lexical[index] = 0.25 * len(query_terms & text_terms) / len(query_terms)
                lexical[index] += 0.15 * len(query_terms & title_terms) / len(query_terms)
        scores = dense_scores + lexical
        ordered = np.argsort(scores)[::-1]
        results: list[RetrievedChunk] = []
        per_document: dict[str, int] = {}
        for raw_index in ordered:
            index = int(raw_index)
            row = self.frame.iloc[index]
            document_id = str(row.document_id)
            if per_document.get(document_id, 0) >= 2:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(row.chunk_id),
                    document_id=document_id,
                    title=str(row.title),
                    source_url=str(row.source_url),
                    text=str(row.text),
                    score=float(scores[index]),
                )
            )
            per_document[document_id] = per_document.get(document_id, 0) + 1
            if len(results) >= top_k:
                break
        return results


def format_retrieval_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(f"[S{index}] {chunk.title}\nURL: {chunk.source_url}\nExcerpt: {chunk.text}")
    return "\n\n".join(sections)
