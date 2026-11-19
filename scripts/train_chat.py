from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint
from cyberslm.training.evidence import AnswerOnlyDataset, EvidenceExample, pad_answer_only_batch
from cyberslm.training.trainer import TrainConfig, Trainer
from cyberslm.utils.config import dataclass_from_yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def examples(path: Path) -> list[EvidenceExample]:
    frame = pd.read_parquet(path)
    return [EvidenceExample(row.prompt, row.answer) for row in frame.itertuples(index=False)]


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant-only SFT for CyberSLM-35M-Chat")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/cyberslm_35m_chat.yaml"),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    args = parser.parse_args()

    validation_report = json.loads(
        (args.dataset_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    dataset_manifest = json.loads(
        (args.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if validation_report["status"] != "PASS":
        raise ValueError("Chat dataset validation report is not PASS")
    tokenizer = ByteTokenizer()
    model_config = load_model_config(str(args.model_config))
    train_config = dataclass_from_yaml(TrainConfig, args.training_config)
    if train_config.sequence_length != model_config.max_seq_len:
        raise ValueError("Training and model sequence lengths differ")
    model = CyberSLM(model_config)
    initial_state = load_checkpoint(
        args.initial_checkpoint,
        model=model,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    train_data = AnswerOnlyDataset(
        examples(args.dataset_dir / "train_chat.parquet"),
        tokenizer=tokenizer,
        max_length=model_config.max_seq_len,
    )
    validation_data = AnswerOnlyDataset(
        examples(args.dataset_dir / "validation_chat.parquet"),
        tokenizer=tokenizer,
        max_length=model_config.max_seq_len,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=pad_answer_only_batch,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=pad_answer_only_batch,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model,
        train_config,
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=args.output_dir / "checkpoints",
        log_path=args.output_dir / "metrics.jsonl",
    )
    iterator = iter(train_loader)
    started = time.perf_counter()
    losses: list[float] = []
    answer_token_exposures = 0
    best_validation = math.inf
    best_step = 0
    stale_evaluations = 0
    while trainer.state.step < train_config.max_steps:
        accumulated = []
        for _ in range(train_config.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            accumulated.append(batch)
            answer_token_exposures += int((batch[1] != -100).sum())
        losses.append(trainer.train_step(accumulated))
        step = trainer.state.step
        if step % train_config.validate_every == 0:
            current = trainer.evaluate(validation_loader)["validation_loss"]
            if current < best_validation:
                best_validation = current
                best_step = step
                stale_evaluations = 0
                trainer.save("best.pt")
            else:
                stale_evaluations += 1
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": losses[-1],
                        "validation_loss": current,
                        "answer_token_exposures": answer_token_exposures,
                    }
                ),
                flush=True,
            )
            if stale_evaluations >= args.early_stopping_patience:
                break
        if step % train_config.checkpoint_every == 0:
            trainer.save(f"step_{step:06d}.pt")
    final_checkpoint = trainer.save(f"final_step_{trainer.state.step:06d}.pt")
    best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
    if not best_checkpoint.exists():
        raise RuntimeError("No best validation checkpoint was created")
    selected = CyberSLM(model_config)
    selected_state = load_checkpoint(
        best_checkpoint,
        model=selected,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    summary = {
        "status": "COMPLETED_BOUNDED_CHAT_SFT",
        "limitation": (
            "This small model was trained on project-authored synthetic conversations; "
            "fluency, factuality, and general reasoning require separate evaluation."
        ),
        "parameter_count": count_parameters(model),
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        "initial_checkpoint_state": initial_state,
        "optimizer_steps": trainer.state.step,
        "answer_token_exposures": answer_token_exposures,
        "training_examples": len(train_data),
        "validation_examples": len(validation_data),
        "effective_example_epochs": (
            trainer.state.step
            * train_config.gradient_accumulation_steps
            * train_config.batch_size
            / len(train_data)
        ),
        "first_training_loss": losses[0],
        "last_training_loss": losses[-1],
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if trainer.device.type == "cuda" else None
        ),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "best_checkpoint_restore_step": selected_state.get("step"),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "dataset_manifest_sha256": sha256_file(args.dataset_dir / "dataset_manifest.json"),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "role_format": dataset_manifest["role_format"],
        "model_config": asdict(model_config),
        "training_config": asdict(train_config),
        "code_revision_at_start": git_revision(),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
