from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cyberslm.models.transformer import ModelConfig


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    state: dict[str, Any],
    model_config: ModelConfig,
    tokenizer_fingerprint: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
            "model_config": asdict(model_config),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "format_version": 1,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    expected_model_config: ModelConfig | None = None,
    expected_tokenizer_fingerprint: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    if expected_model_config is not None and payload["model_config"] != asdict(
        expected_model_config
    ):
        raise ValueError("Checkpoint model configuration is incompatible")
    if (
        expected_tokenizer_fingerprint is not None
        and payload["tokenizer_fingerprint"] != expected_tokenizer_fingerprint
    ):
        raise ValueError("Checkpoint tokenizer fingerprint is incompatible")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    state = payload.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint state")
    return state
