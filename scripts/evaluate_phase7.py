from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic and majority QA baselines")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--phase3-result", type=Path)
    return parser.parse_args()


def exact_match(predictions: list[str], answers: list[str]) -> float:
    return sum(
        prediction.strip() == answer.strip()
        for prediction, answer in zip(predictions, answers, strict=True)
    ) / max(1, len(answers))


def numeric_match(prediction: str, answer: str) -> bool:
    pattern = r"[-+]?\d+(?:\.\d+)?"
    return re.findall(pattern, prediction) == re.findall(pattern, answer)


def unsupported_claim_rate(predictions: list[str]) -> float:
    unsupported = 0
    for prediction in predictions:
        lowered = prediction.lower()
        positive = "confirmed attack" in lowered or "confirmed apt attack" in lowered
        denied = any(term in lowered for term in ("no.", "not ", "cannot", "do not"))
        unsupported += positive and not denied
    return unsupported / max(1, len(predictions))


def evaluate_predictions(predictions: list[str], answers: list[str]) -> dict[str, float]:
    counts = Counter(predictions)
    return {
        "exact_match": exact_match(predictions, answers),
        "numeric_correctness": sum(
            numeric_match(prediction, answer)
            for prediction, answer in zip(predictions, answers, strict=True)
        )
        / max(1, len(answers)),
        "unsupported_claim_rate": unsupported_claim_rate(predictions),
        "answer_repetition_rate": max(counts.values(), default=0) / max(1, len(predictions)),
    }


def main() -> None:
    args = parse_args()
    train = pd.read_parquet(args.dataset_dir / "train_qa.parquet")
    majority = (
        train.groupby("qa_type")["answer"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .to_dict()
    )
    results: dict[str, Any] = {
        "dataset_version": json.loads(
            (args.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )["dataset_version"],
        "baselines": {},
        "models": {},
        "cautions": [
            "The deterministic recomputation baseline is an oracle for these calculated tasks.",
            "The Phase 3 4M next-byte metrics are not directly comparable to answer exact match.",
            "The 35M smoke checkpoint is not a trained evaluation model.",
        ],
    }
    for split in ("validation", "cross_scenario_test"):
        frame = pd.read_parquet(args.dataset_dir / f"{split}_qa.parquet")
        answers = frame["answer"].tolist()
        majority_predictions = [majority.get(task, "") for task in frame["qa_type"]]
        results["baselines"][split] = {
            "examples": len(frame),
            "deterministic_recomputation": evaluate_predictions(answers, answers),
            "majority_answer_by_task": evaluate_predictions(majority_predictions, answers),
        }
    if args.smoke_result and args.smoke_result.exists():
        smoke = json.loads(args.smoke_result.read_text(encoding="utf-8"))
        results["models"]["CyberSLM-35M"] = {
            "status": "engineering_smoke_only_not_evaluated",
            "parameter_count": smoke["parameter_count"],
            "smoke_steps": smoke["smoke_steps_completed"],
            "first_loss": smoke["first_training_loss"],
            "last_loss": smoke["last_training_loss"],
            "peak_cuda_memory_bytes": smoke["peak_cuda_memory_bytes"],
            "checkpoint_reload_step": smoke["checkpoint_reload_step"],
        }
    if args.phase3_result and args.phase3_result.exists():
        results["models"]["CyberSLM-4M"] = {
            "status": "historical_next_byte_evaluation_different_objective",
            "recorded_result": json.loads(args.phase3_result.read_text(encoding="utf-8")),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
