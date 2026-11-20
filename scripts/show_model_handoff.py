"""Backend-only CyberSLM -> verification -> Qwen demonstration.

Run from repository root with PYTHONPATH=src. Does not import Streamlit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cyberslm.two_model_bridge import run_bridge

FICTIONAL_EVENTS: list[dict[str, Any]] = [
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-1",
        "timestamp": 1.0,
    },
    {
        "event_type": "EVENT_WRITE",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-2",
        "timestamp": 2.0,
    },
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-b",
        "destination_entity": "fictional-file-1",
        "timestamp": 3.0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live two-model backend communication demonstration"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "lm_studio"],
        default="ollama",
        help="Qwen runtime; original 35M runs through Ollama",
    )
    parser.add_argument("--question", default="count unique event types")
    parser.add_argument("--events", type=Path, help="Optional local fictional event JSON array")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional destination to save the model communication receipt JSON",
    )
    args = parser.parse_args()

    events = (
        json.loads(args.events.read_text(encoding="utf-8")) if args.events else FICTIONAL_EVENTS
    )
    if not isinstance(events, list):
        parser.error("The --events file must contain a JSON list")

    print("\n[BACKEND] CyberSLM-35M -> independent verification -> Qwen", flush=True)
    print(f"[INPUT] Question: {args.question}", flush=True)
    print(f"[INPUT] Fictional events: {len(events)}", flush=True)

    report = run_bridge(events, args.question, provider=args.provider, retriever=None)

    if report.handoff_packet is None:
        print("[ROUTER] Unsupported question. No neural handoff performed.")
        print(f"[VERIFICATION] {report.verification}")
    else:
        print(f"[1 CYBERSLM] Model: {report.scratch_model}")
        print(f"[1 CYBERSLM] Neural prediction (not verified): {report.scratch_prediction!r}")
        print(
            f"[2 VERIFIER] {report.verification}; "
            f"reference: {report.verified_answer!r}; agrees: {report.agrees}"
        )
        print("[3 HANDOFF PACKET] Actual data passed to Qwen:")
        print(json.dumps(report.handoff_packet, indent=2, ensure_ascii=False))
        print(
            f"[4 QWEN] Provider: {report.explanation_provider}; model: {report.explanation_model}"
        )
    print(f"[FINAL ANSWER] {report.explanation}")
    if report.grounding_guard_note:
        print(f"[GROUNDING NOTE] {report.grounding_guard_note}")

    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[RECEIPT] Saved to {args.receipt}")


if __name__ == "__main__":
    main()
