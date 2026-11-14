from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from cyberslm.models.transformer import CyberSLM
from cyberslm.training.checkpoint import load_checkpoint, save_checkpoint
from cyberslm.utils.reproducibility import set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    batch_size: int = 8
    sequence_length: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    gradient_accumulation_steps: int = 8
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    validate_every: int = 500
    checkpoint_every: int = 1_000
    log_every: int = 10


@dataclass
class TrainingState:
    step: int = 0
    tokens_seen: int = 0
    best_validation_loss: float = math.inf


class Trainer:
    def __init__(
        self,
        model: CyberSLM,
        config: TrainConfig,
        *,
        tokenizer_fingerprint: str,
        checkpoint_dir: str | Path,
        log_path: str | Path,
    ) -> None:
        set_seed(config.seed)
        self.model = model
        self.config = config
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = LambdaLR(self.optimizer, self._lr_multiplier)
        amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.state = TrainingState()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_path = Path(log_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "training_config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device

    def _lr_multiplier(self, step: int) -> float:
        if self.config.warmup_steps and step < self.config.warmup_steps:
            return max(1e-8, (step + 1) / self.config.warmup_steps)
        denominator = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, (step - self.config.warmup_steps) / denominator)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _move_batch(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        return batch[0].to(self.device), batch[1].to(self.device)

    def train_step(self, batches: Iterable[tuple[Tensor, Tensor]]) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        for raw_batch in islice(batches, self.config.gradient_accumulation_steps):
            count += 1
            inputs, labels = self._move_batch(raw_batch)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                output = self.model(inputs, labels)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError("Non-finite training loss")
                loss = output.loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            total_loss += float(output.loss.detach())
            self.state.tokens_seen += labels.numel()
        if count == 0:
            raise ValueError("No batches were provided")
        self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad))):
                raise FloatingPointError("Non-finite gradient detected")
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.state.step += 1
        mean_loss = total_loss / count
        if self.state.step % self.config.log_every == 0:
            self.log_metrics({"train_loss": mean_loss, "perplexity": safe_perplexity(mean_loss)})
        return mean_loss

    @torch.inference_mode()
    def evaluate(self, batches: Iterable[tuple[Tensor, Tensor]]) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        for raw_batch in batches:
            inputs, labels = self._move_batch(raw_batch)
            output = self.model(inputs, labels)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("Non-finite validation loss")
            losses.append(float(output.loss))
        if not losses:
            raise ValueError("No validation batches were provided")
        loss = sum(losses) / len(losses)
        self.state.best_validation_loss = min(self.state.best_validation_loss, loss)
        metrics = {"validation_loss": loss, "validation_perplexity": safe_perplexity(loss)}
        self.log_metrics(metrics)
        return metrics

    def log_metrics(self, metrics: dict[str, float]) -> None:
        record: dict[str, Any] = {
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **metrics,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save(self, name: str | None = None) -> Path:
        path = self.checkpoint_dir / (name or f"step_{self.state.step:08d}.pt")
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=asdict(self.state),
            model_config=self.model.config,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        return path

    def resume(self, path: str | Path) -> None:
        raw_state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_model_config=self.model.config,
            expected_tokenizer_fingerprint=self.tokenizer_fingerprint,
            map_location=self.device,
        )
        self.state = TrainingState(**raw_state)


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 80.0))
