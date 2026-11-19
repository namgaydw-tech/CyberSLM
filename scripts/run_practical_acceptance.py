from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cyberslm.assistant import practical_chat
from cyberslm.rag import KnowledgeRetriever

PROMPTS = [
    ("Hi CyberSLM.", ["hello", "help"]),
    ("Explain ransomware in detail.", ["encrypt", "backup", "incident"]),
    ("How can machine learning help detect it?", ["behavior", "anomal", "false positive"]),
    ("Give me a Python cybersecurity project idea.", ["Python", "authorized", "data"]),
    ("Explain how to implement that project.", ["step", "train", "evaluat"]),
    ("Summarize our discussion.", ["ransomware", "machine learning", "project"]),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run practical CyberSLM acceptance dialogue")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-q4_K_M")
    parser.add_argument("--provider", choices=("ollama", "lm_studio"), default="ollama")
    parser.add_argument("--rag-index", type=Path)
    parser.add_argument("--answer-style", default="Detailed")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    retriever = KnowledgeRetriever(args.rag_index) if args.rag_index else None
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    for turn, (user, terms) in enumerate(PROMPTS, start=1):
        response = practical_chat(
            user,
            history,
            model=args.model,
            provider=args.provider,
            answer_style=args.answer_style,
            retriever=retriever,
            max_tokens=args.max_tokens,
        )
        matched = [term for term in terms if term.lower() in response.content.lower()]
        turn_record = {
            "turn": turn,
            "user": user,
            "assistant": response.content,
            "rubric_terms": terms,
            "matched_terms": matched,
            "rubric_coverage": len(matched) / len(terms),
            "latency_seconds": response.latency_seconds,
            "response_origin": response.response_origin,
            "sources": [
                {
                    "title": source.title,
                    "url": source.source_url,
                    "score": source.score,
                }
                for source in response.sources
            ],
        }
        turns.append(turn_record)
        history.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": response.content},
            ]
        )
        print(json.dumps(turn_record), flush=True)
    report = {
        "status": "COMPLETED",
        "provider": args.provider,
        "model": args.model,
        "rag_enabled": retriever is not None,
        "rag_manifest_sha256": (
            sha256_file(args.rag_index / "manifest.json") if args.rag_index else None
        ),
        "turns": turns,
        "mean_rubric_coverage": sum(item["rubric_coverage"] for item in turns) / len(turns),
        "mean_latency_seconds": sum(item["latency_seconds"] for item in turns) / len(turns),
        "human_review_status": "PENDING_INDEPENDENT_HUMAN_REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
