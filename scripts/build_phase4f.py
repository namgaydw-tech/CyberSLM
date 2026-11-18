from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberslm.data.phase4f import Phase4FConfig, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Phase 4F evidence QA v2")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--session-gap-seconds", type=float, default=30.0)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--answer-limit", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Phase4FConfig(
        seed=args.seed,
        session_gap_seconds=args.session_gap_seconds,
        max_events_per_window=args.max_events,
        min_events_per_window=args.min_events,
        repeated_answer_limit=args.answer_limit,
    )
    manifest = write_dataset(args.source, args.output, config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
