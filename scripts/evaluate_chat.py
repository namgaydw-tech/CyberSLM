from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cyberslm.chat import clean_chat_completion
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


def repeated_phrase(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3))]
    return bool(grams) and max(Counter(grams).values()) >= 3


def number_consistent(output: str, reference: str) -> bool:
    output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", output))
    reference_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", reference))
    return not output_numbers or output_numbers.issubset(reference_numbers)


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CyberSLM chat on unseen scenarios")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-per-category", type=int)
    args = parser.parse_args()
    tokenizer = ByteTokenizer()
    config = load_model_config("configs/model/cyberslm_35m.yaml")
    model = CyberSLM(config)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    frame = pd.read_parquet(args.test)
    if args.max_per_category is not None:
        frame = (
            frame.groupby("category", sort=True).head(args.max_per_category).reset_index(drop=True)
        )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        prompt_ids = tokenizer.encode(row.prompt, bos=True)
        limit = min(args.max_new_tokens, len(row.answer.encode("utf-8")) + 32)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(inputs, max_new_tokens=limit, top_k=1, eos_id=tokenizer.eos_id)[
            0, len(prompt_ids) :
        ].tolist()
        eos_terminated = tokenizer.eos_id in generated
        if eos_terminated:
            generated = generated[: generated.index(tokenizer.eos_id)]
        raw = tokenizer.decode(generated)
        end_terminated = "<|end|>" in raw
        output = clean_chat_completion(raw)
        reference = clean_chat_completion(row.answer)
        terms = json.loads(row.rubric_terms_json)
        matched = [term for term in terms if str(term).lower() in output.lower()]
        coverage = len(matched) / len(terms) if terms else 1.0
        no_repetition = not repeated_phrase(output)
        relevant = bool(output) and coverage >= 0.5
        factual_proxy = (
            number_consistent(output, reference)
            if row.category == "structured_evidence"
            else coverage >= 0.5
        )
        rows.append(
            {
                "qa_id": row.qa_id,
                "conversation_id": row.conversation_id,
                "category": row.category,
                "prompt": row.prompt,
                "reference_answer": reference,
                "model_output": output,
                "raw_model_output": raw,
                "exact_match": output == reference,
                "rubric_coverage": coverage,
                "relevance_proxy": relevant,
                "factuality_proxy": factual_proxy,
                "no_repetition": no_repetition,
                "terminated": eos_terminated or end_terminated,
                "matched_terms_json": json.dumps(matched),
            }
        )
        if index % 12 == 0:
            print(json.dumps({"evaluated": index, "total": len(frame)}), flush=True)
    predictions = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    by_category: dict[str, Any] = {}
    for category, group in predictions.groupby("category", sort=True):
        by_category[str(category)] = {
            "examples": len(group),
            "exact_match": float(group["exact_match"].mean()),
            "mean_rubric_coverage": float(group["rubric_coverage"].mean()),
            "relevance_proxy": float(group["relevance_proxy"].mean()),
            "factuality_proxy": float(group["factuality_proxy"].mean()),
            "no_repetition": float(group["no_repetition"].mean()),
            "termination": float(group["terminated"].mean()),
        }
    exact = int(predictions["exact_match"].sum())
    relevant = int(predictions["relevance_proxy"].sum())
    quality_pass = (
        len(predictions) >= 100
        and relevant / len(predictions) >= 0.6
        and float(predictions["factuality_proxy"].mean()) >= 0.6
        and float(predictions["termination"].mean()) >= 0.9
    )
    report = {
        "status": "PASS" if quality_pass else "FAIL_QUALITY_THRESHOLDS",
        "evaluation_execution": "PASS" if len(predictions) >= 100 else "FAIL",
        "quality_thresholds": {
            "minimum_examples": 100,
            "minimum_relevance_proxy": 0.6,
            "minimum_factuality_proxy": 0.6,
            "minimum_termination": 0.9,
        },
        "examples": len(predictions),
        "unique_unseen_scenarios": int(predictions["conversation_id"].nunique()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": state["step"],
        "test_sha256": sha256_file(args.test),
        "greedy_decoding": True,
        "exact_match": exact / len(predictions),
        "exact_match_wilson_95": wilson(exact, len(predictions)),
        "mean_rubric_coverage": float(predictions["rubric_coverage"].mean()),
        "relevance_proxy": relevant / len(predictions),
        "factuality_proxy": float(predictions["factuality_proxy"].mean()),
        "no_repetition": float(predictions["no_repetition"].mean()),
        "termination": float(predictions["terminated"].mean()),
        "by_category": by_category,
        "duration_seconds": time.perf_counter() - started,
        "human_review_status": "PENDING_INDEPENDENT_HUMAN_REVIEW",
        "metric_limitations": (
            "Keyword, number-consistency, relevance, and repetition checks are automated "
            "proxies. They are not substitutes for factual expert or human quality review."
        ),
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    review = predictions.groupby("category", sort=True).head(2).copy()
    review["human_relevance_1_to_5"] = ""
    review["human_correctness_1_to_5"] = ""
    review["human_fluency_1_to_5"] = ""
    review["human_notes"] = ""
    review.to_csv(args.output_dir / "human_review_packet.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
