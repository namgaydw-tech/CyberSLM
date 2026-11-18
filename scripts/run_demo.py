from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberslm.demo import analyze_structured_events, deterministic_answer

FICTIONAL_EVENTS = [
    {
        "event_type": "EVENT_CONNECT",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-service-1",
        "timestamp": 1.0,
    },
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-1",
        "timestamp": 2.5,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CyberSLM evidence demonstration")
    parser.add_argument("--events-json", type=Path)
    parser.add_argument("--question", default="Which event type is most frequent?")
    args = parser.parse_args()
    events = (
        json.loads(args.events_json.read_text(encoding="utf-8"))
        if args.events_json
        else FICTIONAL_EVENTS
    )
    facts = analyze_structured_events(events)
    output = {
        "input_status": "user_supplied" if args.events_json else "unmistakably_fictional_fixture",
        "observed_facts": facts,
        "question": args.question,
        "answer": deterministic_answer(facts, args.question),
        "model_status": "deterministic baseline; trained CyberSLM checkpoint not selected",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
