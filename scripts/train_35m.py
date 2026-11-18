from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyberslm.data.phase4f import sha256_file
from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint
from cyberslm.training.evidence import (
    AnswerOnlyDataset,
    EvidenceExample,
    pad_answer_only_batch,
)
from cyberslm.training.trainer import TrainConfig, Trainer
from cyberslm.utils.config import dataclass_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CyberSLM-35M from random initialization")
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/cyberslm_35m_pilot.yaml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cyberslm_35m_pilot"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def examples_from_parquet(path: Path) -> list[EvidenceExample]:
    frame = pd.read_parquet(path)
    return [EvidenceExample(row.prompt, row.answer) for row in frame.itertuples(index=False)]


def main() -> None:
    args = parse_args()
    model_config = load_model_config(str(args.model_config))
    training_config = dataclass_from_yaml(TrainConfig, args.training_config)
    tokenizer = ByteTokenizer()
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary and byte tokenizer vocabulary differ")
    model = CyberSLM(model_config)
    parameter_count = count_parameters(model)
    delta_percent = 100 * abs(parameter_count - 35_000_000) / 35_000_000
    if delta_percent > 1.0:
        raise ValueError(f"Parameter count {parameter_count:,} is more than 1% from 35M")

    result: dict[str, object] = {
        "model": "CyberSLM-35M",
        "parameter_count": parameter_count,
        "target_delta_percent": delta_percent,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "model_config_sha256": sha256_file(args.model_config),
        "training_config_sha256": sha256_file(args.training_config),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if args.dataset_dir is None:
        synthetic = [
            EvidenceExample(
                "Recorded-event evidence: EVENT_CONNECT=3, EVENT_READ=1.\nQuestion: "
                "How many EVENT_READ events?\nAnswer: ",
                "There is 1 recorded EVENT_READ event.",
            )
        ]
        train_examples = synthetic
        validation_examples = synthetic
        result["data_status"] = "fictional_smoke_fixture"
    else:
        validation_report = json.loads(
            (args.dataset_dir / "validation_report.json").read_text(encoding="utf-8")
        )
        if validation_report["status"] != "PASS":
            raise ValueError("Phase 4F validation report is not PASS")
        train_examples = examples_from_parquet(args.dataset_dir / "train_qa.parquet")
        validation_examples = examples_from_parquet(args.dataset_dir / "validation_qa.parquet")
        result["data_status"] = "validated_phase4f"
        result["dataset_version"] = validation_report["dataset_version"]
        result["dataset_manifest_sha256"] = sha256_file(args.dataset_dir / "dataset_manifest.json")

    train_data = AnswerOnlyDataset(
        train_examples, tokenizer=tokenizer, max_length=model_config.max_seq_len
    )
    validation_data = AnswerOnlyDataset(
        validation_examples, tokenizer=tokenizer, max_length=model_config.max_seq_len
    )
    max_train_length = max(train_data[index][0].numel() for index in range(len(train_data)))
    result["train_examples"] = len(train_data)
    result["validation_examples"] = len(validation_data)
    result["maximum_train_input_tokens"] = max_train_length
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / "preflight.json"
    preflight_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.check_only or args.smoke_steps <= 0:
        return

    train_loader = DataLoader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=pad_answer_only_batch,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=pad_answer_only_batch,
    )
    trainer = Trainer(
        model,
        training_config,
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=args.output_dir / "checkpoints",
        log_path=args.output_dir / "metrics.jsonl",
    )
    iterator = iter(train_loader)
    losses: list[float] = []
    for _ in range(args.smoke_steps):
        accumulated = []
        for _ in range(training_config.gradient_accumulation_steps):
            try:
                accumulated.append(next(iterator))
            except StopIteration:
                iterator = iter(train_loader)
                accumulated.append(next(iterator))
        losses.append(trainer.train_step(accumulated))
    metrics = trainer.evaluate([next(iter(validation_loader))])
    checkpoint = trainer.save("smoke_checkpoint.pt")
    restored = CyberSLM(model_config)
    restored_state = load_checkpoint(
        checkpoint,
        model=restored,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    generated_ids = restored.generate(
        torch.tensor([[tokenizer.bos_id]]), max_new_tokens=2, top_k=1
    ).tolist()[0]
    result.update(
        {
            "smoke_steps_completed": args.smoke_steps,
            "first_training_loss": losses[0],
            "last_training_loss": losses[-1],
            "validation": metrics,
            "checkpoint": str(checkpoint),
            "checkpoint_reload_step": restored_state["step"],
            "checkpoint_generation_token_ids": generated_ids,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None
            ),
        }
    )
    (args.output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
