from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint
from cyberslm.training.trainer import TrainConfig
from cyberslm.utils.config import dataclass_from_yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a receipt after completed chat SFT")
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
    args = parser.parse_args()
    tokenizer = ByteTokenizer()
    model_config = load_model_config(str(args.model_config))
    train_config = dataclass_from_yaml(TrainConfig, args.training_config)
    best = args.output_dir / "checkpoints" / "best.pt"
    finals = sorted((args.output_dir / "checkpoints").glob("final_step_*.pt"))
    if not best.exists() or not finals:
        raise FileNotFoundError("Completed best and final checkpoints are required")
    final = finals[-1]
    best_model = CyberSLM(model_config)
    best_state = load_checkpoint(
        best,
        model=best_model,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    final_model = CyberSLM(model_config)
    final_state = load_checkpoint(
        final,
        model=final_model,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    initial_model = CyberSLM(model_config)
    initial_state = load_checkpoint(
        args.initial_checkpoint,
        model=initial_model,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    records = [
        json.loads(line)
        for line in (args.output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_records = [item for item in records if "train_loss" in item]
    validation_records = [item for item in records if "validation_loss" in item]
    best_validation = min(validation_records, key=lambda item: item["validation_loss"])
    manifest = json.loads((args.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    training_examples = len(pd.read_parquet(args.dataset_dir / "train_chat.parquet"))
    validation_examples = len(pd.read_parquet(args.dataset_dir / "validation_chat.parquet"))
    final_step = int(final_state["step"])
    summary = {
        "status": "COMPLETED_BOUNDED_CHAT_SFT",
        "receipt_recovery_note": (
            "Weights and metrics were saved before the original summary writer encountered a "
            "metadata-key error. This receipt was reconstructed from those saved artifacts."
        ),
        "limitation": (
            "This small model was trained on project-authored synthetic conversations; "
            "fluency, factuality, and general reasoning require separate evaluation."
        ),
        "parameter_count": count_parameters(best_model),
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        "initial_checkpoint_state": initial_state,
        "optimizer_steps": final_step,
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "effective_example_epochs": (
            final_step
            * train_config.gradient_accumulation_steps
            * train_config.batch_size
            / training_examples
        ),
        "first_logged_training_loss": train_records[0]["train_loss"],
        "last_logged_training_loss": train_records[-1]["train_loss"],
        "best_validation_loss": best_validation["validation_loss"],
        "best_step": best_validation["step"],
        "duration_seconds": None,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "best_checkpoint": str(best.resolve()),
        "best_checkpoint_sha256": sha256_file(best),
        "best_checkpoint_restore_step": best_state.get("step"),
        "final_checkpoint": str(final.resolve()),
        "final_checkpoint_sha256": sha256_file(final),
        "final_checkpoint_restore_step": final_state.get("step"),
        "dataset_manifest_sha256": sha256_file(args.dataset_dir / "dataset_manifest.json"),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "role_format": manifest["role_format"],
        "model_config": asdict(model_config),
        "training_config": asdict(train_config),
        "code_revision_at_recovery": git_revision(),
    }
    output = args.output_dir / "training_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
