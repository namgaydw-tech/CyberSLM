from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


def generate(url: str, model: str, prompt: str, limit: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": limit, "num_ctx": 512},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return str(json.load(response)["response"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a transparent Ollama demo set")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cyberslm-35m-structured")
    parser.add_argument("--url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--correct", type=int, default=3)
    parser.add_argument("--total", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_parquet(args.candidates)
    correct: list[dict[str, Any]] = []
    incorrect: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        reference = str(row.answer)
        output = generate(
            args.url, args.model, str(row.prompt), min(32, len(reference.encode()) + 1)
        )
        result = {
            "id": str(row.qa_id),
            "task": str(row.qa_type),
            "prompt": str(row.prompt),
            "reference_answer": reference,
            "ollama_output": output,
            "ollama_exact_match": output == reference,
        }
        attempts.append(result)
        (correct if result["ollama_exact_match"] else incorrect).append(result)
        if len(correct) >= args.correct and len(incorrect) >= args.total - args.correct:
            break
    selected = correct[: args.correct] + incorrect[: args.total - args.correct]
    if len(correct) < args.correct:
        raise RuntimeError(
            f"Only {len(correct)} correct outputs found in {len(attempts)} candidates"
        )
    report = {
        "selection_policy": (
            "First requested number of correct outputs plus first remaining incorrect "
            "outputs in deterministic candidate order; presentation-only, not an accuracy estimate."
        ),
        "candidates_examined": len(attempts),
        "correct_candidates_found": len(correct),
        "selected_examples": len(selected),
        "selected_correct": sum(item["ollama_exact_match"] for item in selected),
        "model": args.model,
        "selected": selected,
        "all_attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
