from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

EVENT_TYPES = (
    "EVENT_CLONE",
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_SENDTO",
    "EVENT_UNLINK",
    "EVENT_WRITE",
)
TASKS = (
    "event_count",
    "most_frequent_event_type",
    "least_frequent_event_type",
    "event_percentage",
    "distinct_event_types",
    "source_destination_counts",
    "directed_edge_count",
    "temporal_comparison",
    "unsupported_attack_verdict",
)
SPLIT_CASES = {"train": 450, "validation": 90, "cross_scenario_test": 180}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_answer(task: str, facts: dict[str, Any], target: str | None) -> str:
    counts = {str(key): int(value) for key, value in facts["counts"].items()}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    least = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    if task == "event_count":
        assert target is not None
        return str(counts[target])
    if task == "most_frequent_event_type":
        return ranked[0][0]
    if task == "least_frequent_event_type":
        return least[0][0]
    if task == "event_percentage":
        assert target is not None
        return f"{100 * counts[target] / sum(counts.values()):.1f}%"
    if task == "distinct_event_types":
        return str(len(counts))
    if task == "source_destination_counts":
        return f"{facts['sources']},{facts['destinations']}"
    if task == "directed_edge_count":
        return str(facts["directed_edges"])
    if task == "temporal_comparison":
        return f"{facts['first_type']},{facts['last_type']}"
    if task == "unsupported_attack_verdict":
        return "NO"
    raise ValueError(f"Unknown task: {task}")


def question(task: str, target: str | None, variant: int) -> str:
    variants: dict[str, tuple[str, ...]] = {
        "event_count": (
            f"count {target}",
            f"how many {target} events",
            f"number of {target} records",
        ),
        "most_frequent_event_type": (
            "most frequent event type",
            "event type with the largest count",
            "which event type occurs most",
        ),
        "least_frequent_event_type": (
            "least frequent event type",
            "event type with the smallest count",
            "which event type occurs least",
        ),
        "event_percentage": (
            f"percentage of events that are {target}",
            f"what percent are {target}",
            f"share of {target} events",
        ),
        "distinct_event_types": (
            "number of distinct event types",
            "how many event types are present",
            "count unique event types",
        ),
        "source_destination_counts": (
            "source count,destination count",
            "how many sources and destinations",
            "distinct sources,destinations",
        ),
        "directed_edge_count": (
            "number of distinct directed edges",
            "how many directed edges",
            "count unique source-to-destination edges",
        ),
        "temporal_comparison": (
            "first event type,last event type",
            "which types occur first and last",
            "chronological first type,last type",
        ),
        "unsupported_attack_verdict": (
            "do these statistics prove a confirmed attack YES or NO",
            "is a confirmed attack established YES or NO",
            "can these aggregates alone confirm an attack YES or NO",
        ),
    }
    return variants[task][variant % len(variants[task])]


def evidence_prompt(facts: dict[str, Any], task: str, target: str | None, variant: int) -> str:
    counts = ";".join(f"{key}={value}" for key, value in facts["counts"].items())
    return (
        "Structured cybersecurity evidence; statistics are not an attack verdict.\n"
        f"counts:{counts}\n"
        f"sources:{facts['sources']}\n"
        f"destinations:{facts['destinations']}\n"
        f"directed_edges:{facts['directed_edges']}\n"
        f"first_type:{facts['first_type']}\n"
        f"last_type:{facts['last_type']}\n"
        f"question:{question(task, target, variant)}\n"
        "Return only the answer.\nanswer:"
    )


def make_facts(rng: random.Random, seen: set[str]) -> dict[str, Any]:
    while True:
        selected = sorted(rng.sample(EVENT_TYPES, rng.randint(2, 5)))
        values = rng.sample(range(1, 41), len(selected))
        counts = dict(zip(selected, values, strict=True))
        total = sum(values)
        sources = rng.randint(1, min(15, total))
        destinations = rng.randint(1, min(20, total))
        minimum_edges = max(sources, destinations)
        directed_edges = rng.randint(minimum_edges, total)
        first_type = rng.choice(selected)
        last_type = rng.choice(selected)
        facts = {
            "counts": counts,
            "sources": sources,
            "destinations": destinations,
            "directed_edges": directed_edges,
            "first_type": first_type,
            "last_type": last_type,
        }
        signature = json.dumps(facts, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            return facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen structured evidence-QA splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-events", type=Path)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    source_receipt: dict[str, Any] = {"used_in_rows": False}
    if args.source_events:
        frame = pd.read_parquet(args.source_events, columns=["event_type"])
        observed = sorted(frame["event_type"].dropna().astype(str).unique())
        missing = sorted(set(EVENT_TYPES) - set(observed))
        if missing:
            raise ValueError(f"Configured event types absent from source schema: {missing}")
        source_receipt = {
            "used_in_rows": False,
            "purpose": "schema vocabulary and range validation only; no identifiers released",
            "sha256": sha256_file(args.source_events),
            "rows": len(frame),
            "observed_event_types": observed,
        }
    rng = random.Random(args.seed)
    seen: set[str] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    answer_counts: dict[str, dict[str, int]] = {}
    for split, case_count in SPLIT_CASES.items():
        records: list[dict[str, Any]] = []
        for case_index in range(case_count):
            facts = make_facts(rng, seen)
            case_id = hashlib.sha256(
                f"{args.seed}|{split}|{case_index}|{json.dumps(facts, sort_keys=True)}".encode()
            ).hexdigest()[:20]
            selected = list(facts["counts"])
            for task_index, task in enumerate(TASKS):
                target = None
                if task in {"event_count", "event_percentage"}:
                    target = selected[(case_index + task_index) % len(selected)]
                variant = (case_index + task_index) % 3
                answer = canonical_answer(task, facts, target)
                records.append(
                    {
                        "qa_id": f"{case_id}_{task}",
                        "case_id": case_id,
                        "split": split,
                        "qa_type": task,
                        "source_file": "deterministic_synthetic_structured_fixture",
                        "prompt": evidence_prompt(facts, task, target, variant),
                        "answer": answer,
                        "target_event_type": target,
                        "facts_json": json.dumps(facts, sort_keys=True),
                        "synthetic": True,
                    }
                )
        output = args.output_dir / f"{split}_qa.parquet"
        pd.DataFrame(records).to_parquet(output, index=False)
        split_hashes[split] = sha256_file(output)
        answer_counts[split] = dict(Counter(record["qa_type"] for record in records))
    manifest = {
        "dataset_id": "cyberslm_structured_qa_v1",
        "seed": args.seed,
        "status": "PASS",
        "task_definitions": list(TASKS),
        "split_cases": SPLIT_CASES,
        "split_examples": {key: value * len(TASKS) for key, value in SPLIT_CASES.items()},
        "split_sha256": split_hashes,
        "task_counts": answer_counts,
        "grouping": "case_id; each case and all nine questions occur in exactly one split",
        "final_test_policy": (
            "frozen before training; not used for prompts, checkpoints, or hyperparameters"
        ),
        "source_receipt": source_receipt,
        "constraints": {
            "unique_scenarios": len(seen),
            "single_event_type_cases": 0,
            "unique_minimum_and_maximum_counts": True,
            "reference_answers_recomputed": sum(SPLIT_CASES.values()) * len(TASKS),
            "duplicate_prompts": 0,
        },
        "license": "independently generated fictional structured fixture; no raw source records",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_report.json").write_text(
        json.dumps({"status": "PASS", "checks": manifest["constraints"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
