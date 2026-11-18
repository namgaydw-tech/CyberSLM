from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint, save_checkpoint
from cyberslm.training.packed import PackedDocumentDataset
from cyberslm.training.trainer import TrainConfig
from cyberslm.utils.config import dataclass_from_yaml
from cyberslm.utils.reproducibility import set_seed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Stage A CyberSLM-35M pretraining")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/cyberslm_35m_stage_a.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validation-batches", type=int, default=32)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def lr_multiplier(step: int, *, warmup: int, maximum: int) -> float:
    if warmup and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(
    model: CyberSLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    maximum_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for index, (inputs, labels) in enumerate(loader):
        if index >= maximum_batches:
            break
        output = model(inputs.to(device), labels.to(device))
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise FloatingPointError("Non-finite validation loss")
        losses.append(float(output.loss))
    if not losses:
        raise ValueError("No validation batches")
    return sum(losses) / len(losses)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    model_config = load_model_config(str(args.model_config))
    train_config = dataclass_from_yaml(TrainConfig, args.training_config)
    maximum_steps = args.max_steps or train_config.max_steps
    tokenizer = ByteTokenizer()
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    frame = pd.read_parquet(args.corpus)
    required = {"document_id", "split", "text"}
    if required - set(frame.columns):
        raise ValueError(f"Corpus missing columns: {sorted(required - set(frame.columns))}")
    if frame.groupby("document_id")["split"].nunique().max() != 1:
        raise ValueError("document_id crosses splits")
    train_documents = frame[frame["split"] == "train"]["text"].tolist()
    validation_documents = frame[frame["split"] == "validation"]["text"].tolist()
    train_data = PackedDocumentDataset(
        train_documents, tokenizer=tokenizer, sequence_length=train_config.sequence_length
    )
    validation_data = PackedDocumentDataset(
        validation_documents, tokenizer=tokenizer, sequence_length=train_config.sequence_length
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=train_config.batch_size)
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model = CyberSLM(model_config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, warmup=train_config.warmup_steps, maximum=maximum_steps),
    )
    amp_enabled = train_config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    step = 0
    tokens_exposed = 0
    best_validation_loss = math.inf
    best_step = 0
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_model_config=model_config,
            expected_tokenizer_fingerprint=tokenizer.fingerprint,
            map_location=device,
        )
        step = int(state["step"])
        tokens_exposed = int(state["tokens_exposed"])
        best_validation_loss = float(state["best_validation_loss"])
        best_step = int(state["best_step"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(train_loader)
    started = time.perf_counter()
    model.train()
    while step < maximum_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        microbatches = 0
        for _ in range(train_config.gradient_accumulation_steps):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(inputs, labels)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError("Non-finite training loss")
                loss = output.loss / train_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += float(output.loss.detach())
            microbatches += 1
            tokens_exposed += labels.numel()
        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad))):
                raise FloatingPointError("Non-finite gradient")
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        train_loss = accumulated_loss / microbatches
        if step % train_config.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - started
            record = {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
                "tokens_exposed": tokens_exposed,
                "tokens_per_second": tokens_exposed / max(elapsed, 1e-6),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)
        validation_loss: float | None = None
        if step % train_config.validate_every == 0 or step == maximum_steps:
            validation_loss = evaluate(model, validation_loader, device, args.validation_batches)
            model.train()
            append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 80)),
                },
            )
            print(f"validation step={step} loss={validation_loss:.6f}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state={
                        "stage": "A",
                        "step": step,
                        "tokens_exposed": tokens_exposed,
                        "best_validation_loss": best_validation_loss,
                        "best_step": best_step,
                    },
                    model_config=model_config,
                    tokenizer_fingerprint=tokenizer.fingerprint,
                )
        if step % train_config.checkpoint_every == 0 or step == maximum_steps:
            save_checkpoint(
                checkpoint_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state={
                    "stage": "A",
                    "step": step,
                    "tokens_exposed": tokens_exposed,
                    "best_validation_loss": best_validation_loss,
                    "best_step": best_step,
                },
                model_config=model_config,
                tokenizer_fingerprint=tokenizer.fingerprint,
            )

    duration = time.perf_counter() - started
    final_path = checkpoint_dir / f"step_{step:06d}.pt"
    restored = CyberSLM(model_config)
    restored_state = load_checkpoint(
        final_path,
        model=restored,
        expected_model_config=model_config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
        map_location="cpu",
    )
    prompt = "Cybersecurity risk management requires"
    prompt_ids = torch.tensor([tokenizer.encode(prompt, bos=True)], dtype=torch.long)
    generated = restored.generate(prompt_ids, max_new_tokens=48, top_k=1, eos_id=tokenizer.eos_id)
    summary = {
        "status": "COMPLETED_BOUNDED_STAGE_A",
        "objective": "causal next-byte prediction",
        "parameter_count": count_parameters(model),
        "optimizer_steps": step,
        "unique_training_tokens": train_data.unique_token_count,
        "unique_validation_tokens": validation_data.unique_token_count,
        "token_exposures": tokens_exposed,
        "effective_epochs": tokens_exposed / train_data.unique_token_count,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": math.exp(min(best_validation_loss, 80)),
        "best_step": best_step,
        "duration_seconds": duration,
        "tokens_per_second": tokens_exposed / duration,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else None
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "model_config": asdict(model_config),
        "training_config": {**asdict(train_config), "max_steps": maximum_steps},
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "corpus_sha256": sha256_file(args.corpus),
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "model_config_sha256": sha256_file(args.model_config),
        "training_config_sha256": sha256_file(args.training_config),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        ),
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256_file(final_path),
        "checkpoint_restore_step": restored_state["step"],
        "generation_prompt": prompt,
        "generation_output": tokenizer.decode(generated[0].tolist()),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
