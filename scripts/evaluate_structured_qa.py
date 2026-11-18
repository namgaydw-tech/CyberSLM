from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer(task: str, facts: dict[str, Any], target: str | None) -> str:
    counts = {str(key): int(value) for key, value in facts["counts"].items()}
    if task == "event_count":
        assert target is not None
        return str(counts[target])
    if task == "most_frequent_event_type":
        return max(counts, key=lambda key: (counts[key], key))
    if task == "least_frequent_event_type":
        return min(counts, key=lambda key: (counts[key], key))
    if task == "event_percentage":
        assert target is not None
        return f"{100 * counts[target] / sum(counts.values()):.1f}%"
    if task == "distinct_event_types":
        return str(len(counts))
    if task == "source_destination_counts":
        return f"{facts['sources']},{facts['destinations']}"
    if task == "directed_edge_count":
        return str(facts["directed_edges"])
    if task == "temporal_comparison":
        return f"{facts['first_type']},{facts['last_type']}"
    if task == "unsupported_attack_verdict":
        return "NO"
    raise ValueError(task)


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


NUMERIC_TASKS = {
    "event_count",
    "event_percentage",
    "distinct_event_types",
    "source_destination_counts",
    "directed_edge_count",
}


def numeric_value(task: str, value: str) -> int | float | tuple[int, int] | None:
    text = value.strip()
    try:
        if task == "event_percentage":
            return float(text.removesuffix("%"))
        if task == "source_destination_counts":
            match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", text)
            return (int(match.group(1)), int(match.group(2))) if match else None
        if task in NUMERIC_TASKS:
            return int(text)
    except ValueError:
        return None
    return None


def error_class(task: str, prediction: str, reference: str) -> str:
    if prediction == reference:
        return "none"
    if not prediction:
        return "parsing_failure"
    midpoint = len(prediction) // 2
    if midpoint and prediction[:midpoint] == prediction[midpoint:]:
        return "repetition"
    if task in NUMERIC_TASKS:
        return (
            "arithmetic_error" if numeric_value(task, prediction) is not None else "parsing_failure"
        )
    if task in {"most_frequent_event_type", "least_frequent_event_type"} and re.fullmatch(
        r"EVENT_[A-Z]+", prediction
    ):
        return "wrong_task_selection"
    if task == "unsupported_attack_verdict" and prediction in {"YES", "NO"}:
        return "wrong_task_selection"
    return "hallucination"


def evidence_and_question(prompt: str) -> tuple[str, str]:
    match = re.search(r"(?im)^question:\s*(.+)$", prompt)
    if not match:
        return prompt, ""
    return prompt[: match.start()].strip(), match.group(1).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neural and deterministic structured QA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    tokenizer = ByteTokenizer()
    config = load_model_config("configs/model/cyberslm_35m.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberSLM(config)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    model.to(device).eval()
    frame = pd.read_parquet(args.dataset)
    started = time.perf_counter()
    encoded: list[list[int]] = [
        tokenizer.encode(prompt, bos=True) for prompt in frame["prompt"].tolist()
    ]
    task_limits = {
        str(task): min(
            args.max_new_tokens,
            max(len(str(value).encode("utf-8")) for value in group["answer"]) + 1,
        )
        for task, group in frame.groupby("qa_type", sort=False)
    }
    length_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, prompt_ids in enumerate(encoded):
        task = str(frame.iloc[index]["qa_type"])
        length_groups[(len(prompt_ids), task_limits[task])].append(index)
    neural_outputs: dict[int, str] = {}
    for (prompt_length, generation_limit), indices in sorted(length_groups.items()):
        for offset in range(0, len(indices), args.batch_size):
            batch_indices = indices[offset : offset + args.batch_size]
            inputs = torch.tensor(
                [encoded[index] for index in batch_indices], dtype=torch.long, device=device
            )
            generated_batch = model.generate(
                inputs,
                max_new_tokens=min(generation_limit, config.max_seq_len - prompt_length),
                top_k=1,
                eos_id=tokenizer.eos_id,
            )[:, prompt_length:]
            for index, generated_tensor in zip(batch_indices, generated_batch, strict=True):
                generated = generated_tensor.tolist()
                if tokenizer.eos_id in generated:
                    generated = generated[: generated.index(tokenizer.eos_id)]
                neural_outputs[index] = tokenizer.decode(generated).strip()
    rows: list[dict[str, Any]] = []
    checkpoint_sha256 = sha256_file(args.checkpoint)
    for index, row in enumerate(frame.itertuples(index=False)):
        facts = json.loads(row.facts_json)
        target = row.target_event_type if isinstance(row.target_event_type, str) else None
        recomputed = answer(row.qa_type, facts, target)
        if recomputed != row.answer:
            raise ValueError(f"Reference recomputation failed for {row.qa_id}")
        prediction = neural_outputs[index]
        evidence, question = evidence_and_question(row.prompt)
        reference_numeric = numeric_value(row.qa_type, recomputed)
        prediction_numeric = numeric_value(row.qa_type, prediction)
        neural_numeric_correct = (
            prediction_numeric == reference_numeric if row.qa_type in NUMERIC_TASKS else None
        )
        rows.append(
            {
                "qa_id": row.qa_id,
                "case_id": row.case_id,
                "qa_type": row.qa_type,
                "evidence": evidence,
                "question": question,
                "prompt": row.prompt,
                "reference_answer": recomputed,
                "neural_output": prediction,
                "neural_exact_match": prediction == recomputed,
                "neural_numerical_correctness": neural_numeric_correct,
                "hybrid_output": recomputed,
                "hybrid_exact_match": True,
                "hybrid_numerical_correctness": (True if row.qa_type in NUMERIC_TASKS else None),
                "neural_error_class": error_class(row.qa_type, prediction, recomputed),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": state["step"],
                "inference_mode": "pure_neural_greedy_and_deterministic_hybrid",
            }
        )
    predictions = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    by_task: dict[str, Any] = {}
    for task, group in predictions.groupby("qa_type", sort=True):
        correct = int(group["neural_exact_match"].sum())
        by_task[str(task)] = {
            "examples": len(group),
            "neural_exact_match": correct / len(group),
            "neural_wilson_95": wilson(correct, len(group)),
            "hybrid_exact_match": 1.0,
        }
        if task in NUMERIC_TASKS:
            task_numeric_correct = int(group["neural_numerical_correctness"].sum())
            by_task[str(task)]["neural_numerical_correctness"] = task_numeric_correct / len(group)
    neural_correct = int(predictions["neural_exact_match"].sum())
    numeric_predictions = predictions[predictions["neural_numerical_correctness"].notna()]
    numeric_correct = int(numeric_predictions["neural_numerical_correctness"].sum())
    report = {
        "examples": len(predictions),
        "unique_cases": int(predictions["case_id"].nunique()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": state["step"],
        "neural_exact_match": neural_correct / len(predictions),
        "neural_wilson_95": wilson(neural_correct, len(predictions)),
        "neural_numerical_examples": len(numeric_predictions),
        "neural_numerical_correctness": numeric_correct / len(numeric_predictions),
        "neural_numerical_wilson_95": wilson(numeric_correct, len(numeric_predictions)),
        "hybrid_exact_match": 1.0,
        "hybrid_numerical_correctness": 1.0,
        "hybrid_wilson_95": wilson(len(predictions), len(predictions)),
        "by_task": by_task,
        "neural_error_class_counts": {
            str(key): int(value)
            for key, value in predictions["neural_error_class"].value_counts().items()
        },
        "duration_seconds": time.perf_counter() - started,
        "separation_note": (
            "Hybrid outputs are deterministic recomputation, not neural generations."
        ),
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
