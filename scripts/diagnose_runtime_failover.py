"""Live, non-destructive CyberSLM runtime and failover diagnostic."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cyberslm.demo import analyze_structured_events
from cyberslm.native_inference import NativeCyberSLM
from cyberslm.runtime_manager import (
    ProviderConfig,
    QwenRuntimeManager,
    ScratchRuntimeManager,
)
from cyberslm.two_model_bridge import create_scratch_prompt

EVENTS = [
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-1",
    },
    {
        "event_type": "EVENT_WRITE",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-2",
    },
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-b",
        "destination_entity": "fictional-file-1",
    },
]
QUESTIONS = [
    "count unique event types",
    "count EVENT_READ",
    "is a confirmed attack established YES or NO",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    facts = analyze_structured_events(EVENTS)
    ollama = ScratchRuntimeManager()
    native = ScratchRuntimeManager(
        ollama_url="http://127.0.0.1:1",
        native=NativeCyberSLM(device=args.device),
    )
    comparisons = []
    for question in QUESTIONS:
        prompt = create_scratch_prompt(facts, EVENTS, question)
        ollama_result = ollama.infer(prompt)
        native_result = native.infer(prompt)
        comparisons.append(
            {
                "question": question,
                "ollama": asdict(ollama_result),
                "native": asdict(native_result),
                "exact_match": ollama_result.content == native_result.content,
            }
        )

    qwen = QwenRuntimeManager(
        providers=(
            ProviderConfig("ollama", "http://127.0.0.1:1", "unavailable"),
            ProviderConfig(
                "lm_studio",
                "http://127.0.0.1:1234",
                "cyberslm-practical",
                request_timeout_seconds=90,
            ),
        )
    )
    qwen_result = qwen.chat(
        "In two sentences, explain why anomaly scores do not prove an attack.",
        [],
        max_tokens=96,
        retriever=None,
    )
    report = {
        "scratch_comparisons": comparisons,
        "scratch_all_exact": all(item["exact_match"] for item in comparisons),
        "qwen_failover": {
            "provider": qwen_result.provider,
            "model": qwen_result.model,
            "content": qwen_result.content,
            "attempts": qwen_result.runtime_attempts,
        },
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
