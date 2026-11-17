from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from cyberslm.models.transformer import CyberSLM, count_parameters


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float
    parameter_count: int
    mean_batch_latency_ms: float
    peak_memory_bytes: int | None


@torch.inference_mode()
def evaluate_language_model(
    model: CyberSLM,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: str | torch.device = "cpu",
) -> EvaluationResult:
    target = torch.device(device)
    model.to(target).eval()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    losses: list[float] = []
    latencies: list[float] = []
    for inputs, labels in batches:
        start = time.perf_counter()
        output = model(inputs.to(target), labels.to(target))
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        latencies.append((time.perf_counter() - start) * 1000)
        if output.loss is None:
            raise RuntimeError("Evaluation requires labels")
        losses.append(float(output.loss))
    if not losses:
        raise ValueError("Evaluation requires at least one batch")
    loss = sum(losses) / len(losses)
    peak = torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None
    return EvaluationResult(
        loss=loss,
        perplexity=math.exp(min(loss, 80)),
        parameter_count=count_parameters(model),
        mean_batch_latency_ms=sum(latencies) / len(latencies),
        peak_memory_bytes=peak,
    )
