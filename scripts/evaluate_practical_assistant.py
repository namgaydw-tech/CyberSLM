from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from cyberslm.assistant import practical_chat
from cyberslm.rag import KnowledgeRetriever


def repetition_ratio(text: str) -> float:
    words = re.findall(r"[a-z0-9]+", text.lower())
    trigrams = list(zip(words, words[1:], words[2:], strict=False))
    if not trigrams:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated practical-assistant answers")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-q4_K_M")
    parser.add_argument("--provider", choices=("ollama", "lm_studio"), default="ollama")
    parser.add_argument("--rag-index", type=Path)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    specification = json.loads(args.cases.read_text(encoding="utf-8"))
    retriever = KnowledgeRetriever(args.rag_index) if args.rag_index else None
    records: list[dict[str, Any]] = []
    for case in specification["cases"]:
        response = practical_chat(
            case["prompt"],
            [],
            model=args.model,
            provider=args.provider,
            answer_style="Balanced",
            retriever=retriever,
            max_tokens=args.max_tokens,
        )
        terms = [str(item) for item in case["terms"]]
        matched = [item for item in terms if item.lower() in response.content.lower()]
        record = {
            **case,
            "answer": response.content,
            "matched_terms": matched,
            "term_coverage": len(matched) / len(terms),
            "source_count": len(response.sources),
            "source_requirement_met": not case["needs_sources"] or bool(response.sources),
            "repetition_ratio": repetition_ratio(response.content),
            "latency_seconds": response.latency_seconds,
            "response_origin": response.response_origin,
            "sources": [
                {"title": item.title, "url": item.source_url, "score": item.score}
                for item in response.sources
            ],
        }
        records.append(record)
        summary = {key: record[key] for key in ("id", "term_coverage", "source_count")}
        print(json.dumps(summary), flush=True)

    output = {
        "status": "COMPLETED",
        "model": args.model,
        "provider": args.provider,
        "rag_enabled": retriever is not None,
        "case_count": len(records),
        "mean_term_coverage": sum(item["term_coverage"] for item in records) / len(records),
        "source_requirement_rate": (
            sum(item["source_requirement_met"] for item in records) / len(records)
        ),
        "mean_repetition_ratio": sum(item["repetition_ratio"] for item in records) / len(records),
        "interpretation": (
            "Automated diagnostics only; generated answers require qualitative review."
        ),
        "human_review_status": "PENDING_INDEPENDENT_HUMAN_REVIEW",
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
