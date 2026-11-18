from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint


def numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


def unsupported_positive_claim(text: str) -> bool:
    lowered = text.lower()
    positive = "confirmed attack" in lowered or "confirmed apt attack" in lowered
    denied = any(term in lowered for term in ("no.", "not ", "cannot", "do not"))
    return positive and not denied


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score evidence-QA checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument("--splits", nargs="+", default=["validation", "cross_scenario_test"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    tokenizer = ByteTokenizer()
    config = load_model_config(str(args.model_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberSLM(config)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": state["step"],
        "device": str(device),
        "splits": {},
    }
    for split in args.splits:
        frame = pd.read_parquet(args.dataset_dir / f"{split}_qa.parquet")
        rows: list[dict[str, Any]] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for row in frame.itertuples(index=False):
            prompt_ids = tokenizer.encode(row.prompt, bos=True)
            remaining = max(1, config.max_seq_len - len(prompt_ids))
            maximum = min(args.max_new_tokens, remaining)
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            item_started = time.perf_counter()
            generated = model.generate(
                inputs,
                max_new_tokens=maximum,
                top_k=1,
                eos_id=tokenizer.eos_id,
            )[0, len(prompt_ids) :].tolist()
            if tokenizer.eos_id in generated:
                generated = generated[: generated.index(tokenizer.eos_id)]
            prediction = tokenizer.decode(generated).strip()
            latency = time.perf_counter() - item_started
            rows.append(
                {
                    "split": split,
                    "qa_type": row.qa_type,
                    "source_file": row.source_file,
                    "answer": row.answer,
                    "prediction": prediction,
                    "exact_match": prediction == row.answer.strip(),
                    "numeric_correct": numbers(prediction) == numbers(row.answer),
                    "unsupported_positive_claim": unsupported_positive_claim(prediction),
                    "latency_seconds": latency,
                }
            )
        elapsed = time.perf_counter() - started
        predictions = pd.DataFrame(rows)
        predictions.to_parquet(args.output_dir / f"{split}_predictions.parquet", index=False)
        task_metrics: dict[str, dict[str, float | int]] = {}
        for task, group in predictions.groupby("qa_type", sort=True):
            task_metrics[str(task)] = {
                "examples": len(group),
                "exact_match": float(group["exact_match"].mean()),
                "numeric_correctness": float(group["numeric_correct"].mean()),
            }
        counts = Counter(predictions["prediction"])
        report["splits"][split] = {
            "examples": len(predictions),
            "exact_match": float(predictions["exact_match"].mean()),
            "numeric_correctness": float(predictions["numeric_correct"].mean()),
            "unsupported_claim_rate": float(predictions["unsupported_positive_claim"].mean()),
            "answer_repetition_rate": max(counts.values(), default=0) / max(1, len(predictions)),
            "unique_predictions": predictions["prediction"].nunique(),
            "mean_latency_seconds": float(predictions["latency_seconds"].mean()),
            "total_duration_seconds": elapsed,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
            "by_task": task_metrics,
            "top_predictions": dict(counts.most_common(10)),
        }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
