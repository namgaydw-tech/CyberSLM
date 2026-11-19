from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a longer CyberSLM training context")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=768)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    args = parser.parse_args()
    original = load_model_config(str(args.model_config))
    probed = replace(original, max_seq_len=args.sequence_length, gradient_checkpointing=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CyberSLM(probed)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inputs = torch.randint(3, original.vocab_size, (1, args.sequence_length), device=device)
    labels = inputs.clone()
    amp = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp):
        result = model(inputs, labels)
    if result.loss is None or not bool(torch.isfinite(result.loss)):
        raise RuntimeError("Non-finite context probe loss")
    result.loss.backward()
    report = {
        "status": "PASS",
        "device": str(device),
        "sequence_length": args.sequence_length,
        "forward_backward_loss": float(result.loss.detach()),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else None
        ),
        "decision": "retain_512",
        "decision_reason": (
            "The corpus fits 512 bytes and prior training only validated positions through 512; "
            "a memory probe alone does not establish quality for longer positions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
