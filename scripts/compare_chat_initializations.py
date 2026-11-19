from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint
from cyberslm.training.evidence import AnswerOnlyDataset, EvidenceExample, pad_answer_only_batch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def validation_loss(
    checkpoint: Path, dataset: AnswerOnlyDataset, model_config_path: Path, device: torch.device
) -> float:
    tokenizer = ByteTokenizer()
    config = load_model_config(str(model_config_path))
    model = CyberSLM(config)
    load_checkpoint(
        checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    model.to(device).eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_answer_only_batch)
    for inputs, labels in loader:
        result = model(inputs.to(device), labels.to(device))
        if result.loss is None:
            raise RuntimeError("Model did not return validation loss")
        losses.append(float(result.loss))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CyberSLM chat initial checkpoints")
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.validation)
    tokenizer = ByteTokenizer()
    dataset = AnswerOnlyDataset(
        [EvidenceExample(row.prompt, row.answer) for row in frame.itertuples(index=False)],
        tokenizer=tokenizer,
        max_length=512,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = {"stage_a": args.stage_a, "structured_stage_b": args.structured}
    results = {
        name: {
            "checkpoint": str(path.resolve()),
            "sha256": sha256_file(path),
            "validation_loss": validation_loss(path, dataset, args.model_config, device),
        }
        for name, path in candidates.items()
    }
    selected = min(results, key=lambda item: results[item]["validation_loss"])
    report = {
        "status": "PASS",
        "selection_rule": "lowest assistant-only loss on held-out chat validation split",
        "selected": selected,
        "selected_checkpoint": results[selected]["checkpoint"],
        "validation_examples": len(dataset),
        "device": str(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
