from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from cyberslm.data.phase4f import event_type_entropy, load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 4E source and QA concentration")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.source).set_index("event_uuid", drop=False)
    windows = pd.read_parquet(args.windows)
    qa = pd.concat(
        [
            pd.read_parquet(args.qa_dir / f"{split}_qa.parquet")
            for split in ("train", "validation", "cross_scenario_test")
        ],
        ignore_index=True,
    )
    stats: list[dict[str, Any]] = []
    edge_windows: defaultdict[tuple[str, str], int] = defaultdict(int)
    for window in windows.itertuples(index=False):
        ids = json.loads(window.event_uuids_json)
        chunk = events.loc[ids]
        counts = Counter(chunk["event_type"])
        for edge in set(zip(chunk["source_entity"], chunk["destination_entity"], strict=True)):
            edge_windows[edge] += 1
        stats.append(
            {
                "split": window.split,
                "attack_chain": window.attack_chain,
                "single_type": len(counts) == 1,
                "entropy": event_type_entropy(counts),
                "event_count": len(chunk),
                "duration": float(
                    chunk["event_time_seconds"].max() - chunk["event_time_seconds"].min()
                ),
            }
        )
    frame = pd.DataFrame(stats)
    train_answers = qa[qa["split"] == "train"]["answer"].value_counts()
    result = {
        "phase": "4E source-diversity audit",
        "source_event_rows": len(events),
        "event_type_distribution_by_attack_chain": {
            family: {str(k): int(v) for k, v in Counter(group["event_type"]).most_common()}
            for family, group in events.groupby("attack_chain", sort=True)
        },
        "windows": {
            "count": len(frame),
            "single_event_type_count": int(frame["single_type"].sum()),
            "single_event_type_percentage": 100 * float(frame["single_type"].mean()),
            "mean_entropy_bits": float(frame["entropy"].mean()),
            "median_duration_seconds": float(frame["duration"].median()),
        },
        "entities": {
            "unique_sources": int(events["source_entity"].nunique()),
            "unique_destinations": int(events["destination_entity"].nunique()),
            "top_source_counts": {
                str(k): int(v) for k, v in events["source_entity"].value_counts().head(10).items()
            },
        },
        "edges": {
            "unique": len(edge_windows),
            "reused_across_windows": sum(value > 1 for value in edge_windows.values()),
            "maximum_window_reuse": max(edge_windows.values(), default=0),
        },
        "qa": {
            "examples": len(qa),
            "duplicate_prompt_answer_pairs": int(qa.duplicated(["prompt", "answer"]).sum()),
            "unique_answers": int(qa["answer"].nunique()),
            "top_training_answers": {str(k): int(v) for k, v in train_answers.head(20).items()},
            "top_three_training_answer_share": float(
                train_answers.head(3).sum() / max(1, len(qa[qa["split"] == "train"]))
            ),
        },
        "interpretation": {
            "source_imbalance": "EVENT_CONNECT dominance is present in the original source rows.",
            "windowing_effect": (
                "Random fixed-size sequential chunks propagate that dominance and create "
                "many single-type windows."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
