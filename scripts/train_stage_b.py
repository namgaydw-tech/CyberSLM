from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from cyberslm.training.evidence import (
    AnswerOnlyDataset,
    EvidenceExample,
    pad_answer_only_batch,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Limited-data Stage B evidence adaptation")
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/cyberslm_35m_stage_b.yaml"),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    args = parser.parse_args()

    validation_report = json.loads(
        (args.dataset_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    if validation_report["status"] != "PASS":
        raise ValueError("Phase 4F validation report is not PASS")
    tokenizer = ByteTokenizer()
    model_config = load_model_config(str(args.model_config))
    train_config = dataclass_from_yaml(TrainConfig, args.training_config)
    model = CyberSLM(model_config)
    stage_a_state = load_checkpoint(
        args.stage_a_checkpoint,
        model=model,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    if stage_a_state.get("stage") != "A":
        raise ValueError("Initial checkpoint is not a Stage A checkpoint")
    train_data = AnswerOnlyDataset(
        examples(args.dataset_dir / "train_qa.parquet"),
        tokenizer=tokenizer,
        max_length=model_config.max_seq_len,
    )
    validation_data = AnswerOnlyDataset(
        examples(args.dataset_dir / "validation_qa.parquet"),
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
            metrics = trainer.evaluate(validation_loader)
            current = metrics["validation_loss"]
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
    duration = time.perf_counter() - started
    restored = CyberSLM(model_config)
    restored_trainer = Trainer(
        restored,
        train_config,
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=args.output_dir / "restore_check",
        log_path=args.output_dir / "restore_metrics.jsonl",
    )
    restored_trainer.resume(final_checkpoint)
    restored.to("cpu")
    prompts = [
        (
            "Recorded-event evidence (no attack verdict):\nTotal events: 4\n"
            "Event type counts: EVENT_CONNECT=3, EVENT_READ=1\n\n"
            "Question: How many EVENT_READ events are recorded?\nAnswer: "
        ),
        (
            "Recorded-event evidence (no attack verdict):\nTotal events: 5\n"
            "Event type counts: EVENT_OPEN=2, EVENT_READ=3\n\n"
            "Question: Which event type is most frequent?\nAnswer: "
        ),
        (
            "Recorded-event evidence (no attack verdict):\nTotal events: 3\n"
            "Event type counts: EVENT_CONNECT=3\n\n"
            "Question: Does this prove a confirmed APT attack?\nAnswer: "
        ),
    ]
    restored.eval()
    outputs: list[dict[str, str]] = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt, bos=True)], dtype=torch.long)
        generated = restored.generate(
            input_ids, max_new_tokens=96, top_k=1, eos_id=tokenizer.eos_id
        )[0].tolist()
        completion = tokenizer.decode(generated[len(input_ids[0]) :])
        outputs.append({"prompt": prompt, "output": completion})
    summary = {
        "status": "COMPLETED_STAGE_B_ADAPTATION",
        "limitation": (
            f"Structured adaptation used {len(train_data)} examples; broad cybersecurity "
            "reasoning is not established by task-template performance."
        ),
        "parameter_count": count_parameters(model),
        "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
        "stage_a_checkpoint_sha256": sha256_file(args.stage_a_checkpoint),
        "stage_a_step": stage_a_state["step"],
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
        "duration_seconds": duration,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if trainer.device.type == "cuda" else None
        ),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "checkpoint_restore_step": restored_trainer.state.step,
        "dataset_manifest_sha256": sha256_file(args.dataset_dir / "dataset_manifest.json"),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "model_config": asdict(model_config),
        "training_config": asdict(train_config),
        "generated_examples": outputs,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
