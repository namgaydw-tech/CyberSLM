from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from cyberslm.tokenization.byte import ByteTokenizer

REQUIRED_EVENT_COLUMNS = {
    "event_uuid",
    "event_type",
    "event_time_seconds",
    "source_entity",
    "destination_entity",
    "attack_chain",
}
SPLITS = ("train", "validation", "cross_scenario_test")


@dataclass(frozen=True)
class Phase4FConfig:
    version: str = "phase4f_evidence_qa_v2"
    seed: int = 2026
    session_gap_seconds: float = 30.0
    max_events_per_window: int = 96
    min_events_per_window: int = 8
    validation_fraction: float = 0.20
    max_sequence_tokens: int = 512
    repeated_answer_limit: int = 60
    cross_scenario_prefixes: tuple[str, ...] = ("theia_3.3",)
    train_family: str = "theia_3.11_success"
    task_weights: dict[str, float] = field(
        default_factory=lambda: {
            "event_frequency": 1.0,
            "event_type_comparison": 1.0,
            "event_type_diversity": 0.75,
            "temporal_ordering": 1.0,
            "transition_count": 1.0,
            "entity_relationships": 1.0,
            "directed_graph_statistics": 1.0,
            "event_proportion": 1.0,
            "supported_observation": 0.75,
            "unsupported_claim": 0.5,
            "uncertainty_statement": 0.5,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(*parts: object, seed: int) -> float:
    raw = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path)
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    events = events.copy()
    events["event_uuid"] = events["event_uuid"].astype(str).str.strip()
    if events["event_uuid"].duplicated().any():
        raise ValueError("Source event UUIDs must be unique")
    events["event_type"] = events["event_type"].fillna("UNKNOWN").astype(str)
    events["attack_chain"] = (
        events["attack_chain"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    events["source_entity"] = events["source_entity"].fillna("UNKNOWN_SOURCE").astype(str)
    events["destination_entity"] = (
        events["destination_entity"].fillna("UNKNOWN_DESTINATION").astype(str)
    )
    events["event_time_seconds"] = pd.to_numeric(events["event_time_seconds"], errors="coerce")
    return events


def event_type_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _split_for_session(family: str, source: str, session_key: str, config: Phase4FConfig) -> str:
    if any(family.startswith(prefix) for prefix in config.cross_scenario_prefixes):
        return "cross_scenario_test"
    if family != config.train_family:
        return "quarantine"
    # One source dominates 3.11. Source-exclusive validation would make either
    # train or validation unusable, so natural sessions are split deterministically.
    fraction = stable_fraction(family, source, session_key, seed=config.seed)
    return "validation" if fraction < config.validation_fraction else "train"


def build_windows(events: pd.DataFrame, config: Phase4FConfig) -> pd.DataFrame:
    """Create deterministic source-related temporal sessions without mixing unrelated events."""
    records: list[dict[str, Any]] = []
    grouped = events.sort_values(
        ["attack_chain", "source_entity", "event_time_seconds", "event_uuid"]
    )
    for (family, source), group in grouped.groupby(["attack_chain", "source_entity"], sort=True):
        group = group.reset_index(drop=True)
        session: list[int] = []
        session_number = 0
        previous_time: float | None = None

        def flush(
            indices: list[int],
            *,
            session_group: pd.DataFrame = group,
            session_family: str = str(family),
            session_source: str = str(source),
        ) -> None:
            nonlocal session_number
            if not indices:
                return
            for offset in range(0, len(indices), config.max_events_per_window):
                chunk = session_group.iloc[indices[offset : offset + config.max_events_per_window]]
                ids = chunk["event_uuid"].tolist()
                uuid_json = json.dumps(sorted(ids), separators=(",", ":"))
                window_key = hashlib.sha256(
                    (
                        f"{session_family}|{session_source}|{session_number}|{offset}|{config.seed}"
                    ).encode()
                ).hexdigest()[:16]
                split = _split_for_session(session_family, session_source, window_key, config)
                if len(chunk) < config.min_events_per_window:
                    split = "quarantine"
                times = chunk["event_time_seconds"].dropna()
                records.append(
                    {
                        "window_id": f"session_{window_key}",
                        "split": split,
                        "attack_chain": session_family,
                        "source_group_sha256": hashlib.sha256(session_source.encode()).hexdigest(),
                        "event_count": len(chunk),
                        "event_uuid_sha256": hashlib.sha256(uuid_json.encode()).hexdigest(),
                        "event_uuids_json": uuid_json,
                        "start_time_seconds": float(times.min()) if len(times) else None,
                        "end_time_seconds": float(times.max()) if len(times) else None,
                        "windowing_method": "attack_chain_source_entity_time_gap",
                    }
                )
            session_number += 1

        for index, row in group.iterrows():
            current = row["event_time_seconds"]
            if (
                session
                and previous_time is not None
                and pd.notna(current)
                and current - previous_time > config.session_gap_seconds
            ):
                flush(session)
                session = []
            session.append(index)
            if pd.notna(current):
                previous_time = float(current)
        flush(session)
    return pd.DataFrame.from_records(records)


def _window_facts(chunk: pd.DataFrame) -> dict[str, Any]:
    ordered = chunk.reset_index(drop=True).sort_values(["event_time_seconds", "event_uuid"])
    counts = Counter(ordered["event_type"])
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    edges = list(zip(ordered["source_entity"], ordered["destination_entity"], strict=True))
    edge_counts = Counter(edges)
    types = ordered["event_type"].tolist()
    transitions = sum(left != right for left, right in itertools.pairwise(types))
    times = ordered["event_time_seconds"].dropna()
    return {
        "total": len(ordered),
        "counts": dict(sorted(counts.items())),
        "ranked": ranked,
        "sources": ordered["source_entity"].nunique(),
        "destinations": ordered["destination_entity"].nunique(),
        "distinct_edges": len(edge_counts),
        "repeated_edges": sum(count - 1 for count in edge_counts.values()),
        "first_type": types[0],
        "last_type": types[-1],
        "transitions": transitions,
        "duration": float(times.max() - times.min()) if len(times) > 1 else 0.0,
        "entropy": event_type_entropy(counts),
    }


def _evidence_text(facts: dict[str, Any]) -> str:
    counts = ", ".join(f"{kind}={count}" for kind, count in facts["counts"].items())
    return (
        "Recorded-event evidence (no attack verdict):\n"
        f"Total events: {facts['total']}\n"
        f"Event type counts: {counts}\n"
        f"Chronological first/last types: {facts['first_type']} / {facts['last_type']}\n"
        f"Adjacent type transitions: {facts['transitions']}\n"
        f"Distinct sources/destinations: {facts['sources']} / {facts['destinations']}\n"
        f"Distinct directed edges: {facts['distinct_edges']}\n"
        f"Repeated directed-edge occurrences: {facts['repeated_edges']}\n"
        f"Time span seconds: {facts['duration']:.3f}\n"
    )


def _task_candidates(facts: dict[str, Any]) -> list[tuple[str, str, str]]:
    ranked: list[tuple[str, int]] = facts["ranked"]
    top_type, top_count = ranked[0]
    tasks = [
        (
            "event_frequency",
            f"How many {top_type} events are recorded?",
            f"There are {top_count} recorded {top_type} events.",
        ),
        (
            "event_type_diversity",
            "How many distinct event types are recorded?",
            f"There are {len(ranked)} distinct event types.",
        ),
        (
            "temporal_ordering",
            "Which event types appear first and last chronologically?",
            f"{facts['first_type']} appears first and {facts['last_type']} appears last.",
        ),
        (
            "transition_count",
            "How many adjacent changes in event type occur in chronological order?",
            f"There are {facts['transitions']} adjacent event-type changes.",
        ),
        (
            "entity_relationships",
            "How many distinct source and destination entities are observed?",
            f"The evidence contains {facts['sources']} sources and "
            f"{facts['destinations']} destinations.",
        ),
        (
            "directed_graph_statistics",
            "How many distinct directed edges and repeated edge occurrences are observed?",
            f"There are {facts['distinct_edges']} distinct directed edges and "
            f"{facts['repeated_edges']} repeated edge occurrences.",
        ),
        (
            "event_proportion",
            f"What percentage of events are {top_type}?",
            f"{top_type} accounts for {100 * top_count / facts['total']:.1f}% of events.",
        ),
        (
            "supported_observation",
            "What count-based observation is supported without assigning an attack label?",
            f"The most frequent recorded type is {top_type} with {top_count} of "
            f"{facts['total']} events; this alone is not an attack verdict.",
        ),
        (
            "unsupported_claim",
            "Does this evidence prove that a confirmed APT attack occurred?",
            "No. These event aggregates do not independently prove a confirmed APT attack.",
        ),
        (
            "uncertainty_statement",
            "What important limitation should accompany an interpretation of this window?",
            "The window is a source-related temporal grouping, not a verified incident boundary.",
        ),
    ]
    if len(ranked) >= 2:
        second_type, second_count = ranked[1]
        tasks.append(
            (
                "event_type_comparison",
                f"How many more {top_type} events than {second_type} events are recorded?",
                f"There are {top_count - second_count} more {top_type} events than "
                f"{second_type} events.",
            )
        )
    return tasks


def generate_qa(
    events: pd.DataFrame, windows: pd.DataFrame, config: Phase4FConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    tokenizer = ByteTokenizer()
    indexed = events.set_index("event_uuid", drop=False)
    candidates: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    for window in windows.itertuples(index=False):
        if window.split == "quarantine":
            continue
        ids = json.loads(window.event_uuids_json)
        facts = _window_facts(indexed.loc[ids])
        evidence = _evidence_text(facts)
        facts_json = json.dumps(
            {key: value for key, value in facts.items() if key != "ranked"},
            sort_keys=True,
            separators=(",", ":"),
        )
        for task, question, answer in _task_candidates(facts):
            weight = config.task_weights.get(task, 0.0)
            if weight <= 0 or stable_fraction(window.window_id, task, seed=config.seed) > weight:
                diagnostics["task_weight_filtered"] += 1
                continue
            prompt = f"{evidence}\nQuestion: {question}\nAnswer: "
            token_count = len(tokenizer.encode(prompt + answer, bos=True, eos=True))
            if token_count > config.max_sequence_tokens:
                diagnostics["overlength_filtered"] += 1
                continue
            pair_hash = hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()
            candidates.append(
                {
                    "split": window.split,
                    "source_file": window.window_id,
                    "investigation_family": window.attack_chain,
                    "event_uuid_sha256": window.event_uuid_sha256,
                    "qa_type": task,
                    "prompt": prompt,
                    "answer": answer,
                    "evidence_facts_json": facts_json,
                    "pair_sha256": pair_hash,
                    "token_count": token_count,
                    "source_evidence_status": "RAW_EVENT_AGGREGATE_RECOMPUTABLE",
                    "review_status": "AUTOMATED_RECOMPUTATION_REQUIRED",
                    "training_approved": False,
                }
            )
    candidates.sort(key=lambda row: (row["split"], row["pair_sha256"], row["source_file"]))
    kept: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    answer_counts: Counter[tuple[str, str]] = Counter()
    for row in candidates:
        if row["pair_sha256"] in seen_pairs:
            diagnostics["duplicate_pair_filtered"] += 1
            continue
        key = (row["split"], row["answer"])
        if answer_counts[key] >= config.repeated_answer_limit:
            diagnostics["repeated_answer_filtered"] += 1
            continue
        seen_pairs.add(row["pair_sha256"])
        answer_counts[key] += 1
        row["qa_sha256"] = hashlib.sha256(
            (row["source_file"] + "|" + row["qa_type"] + "|" + row["pair_sha256"]).encode()
        ).hexdigest()
        kept.append(row)
    return pd.DataFrame.from_records(kept), dict(sorted(diagnostics.items()))


def write_dataset(source_path: Path, output_dir: Path, config: Phase4FConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(source_path)
    windows = build_windows(events, config)
    qa, generation_diagnostics = generate_qa(events, windows, config)
    windows_path = output_dir / "event_window_manifest.parquet"
    windows.to_parquet(windows_path, index=False)
    outputs = [windows_path]
    for split in SPLITS:
        path = output_dir / f"{split}_qa.parquet"
        qa[qa["split"] == split].reset_index(drop=True).to_parquet(path, index=False)
        outputs.append(path)
    manifest = {
        "dataset_version": config.version,
        "source": {
            "path_recorded_for_local_reproducibility": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "rows": len(events),
        },
        "configuration": asdict(config),
        "window_counts": windows["split"].value_counts().sort_index().to_dict(),
        "qa_counts": qa["split"].value_counts().sort_index().to_dict(),
        "generation_diagnostics": generation_diagnostics,
        "limitations": [
            "Source attack annotations are not treated as independently verified labels.",
            "A source-related temporal session is not necessarily an incident boundary.",
            "The dominant 3.11 source identity appears in both train and validation; "
            "source-exclusive splitting would make one partition unusable.",
            "Cross-scenario coverage is limited to available THEIA families.",
            "Automated aggregate consistency does not establish cybersecurity reasoning.",
        ],
    }
    receipts = {path.name: sha256_file(path) for path in outputs}
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipts["dataset_manifest.json"] = sha256_file(output_dir / "dataset_manifest.json")
    (output_dir / "sha256_receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def summarize_diversity(
    events: pd.DataFrame, windows: pd.DataFrame, qa: pd.DataFrame
) -> dict[str, Any]:
    indexed = events.set_index("event_uuid", drop=False)
    window_stats: list[dict[str, Any]] = []
    edge_windows: defaultdict[tuple[str, str], int] = defaultdict(int)
    for window in windows.itertuples(index=False):
        chunk = indexed.loc[json.loads(window.event_uuids_json)]
        counts = Counter(chunk["event_type"])
        edges = set(zip(chunk["source_entity"], chunk["destination_entity"], strict=True))
        for edge in edges:
            edge_windows[edge] += 1
        window_stats.append(
            {
                "split": window.split,
                "attack_chain": window.attack_chain,
                "entropy": event_type_entropy(counts),
                "single_type": len(counts) == 1,
                "event_count": len(chunk),
            }
        )
    stats = pd.DataFrame(window_stats)
    answer_counts = qa[qa["split"] == "train"]["answer"].value_counts() if len(qa) else pd.Series()
    return {
        "event_types_by_attack_chain": {
            family: {str(k): int(v) for k, v in Counter(group["event_type"]).most_common()}
            for family, group in events.groupby("attack_chain", sort=True)
        },
        "windows": {
            "count": len(stats),
            "single_event_type_percentage": 100 * float(stats["single_type"].mean()),
            "mean_event_type_entropy_bits": float(stats["entropy"].mean()),
            "by_split": stats.groupby("split")["event_count"]
            .agg(["count", "sum"])
            .to_dict("index"),
        },
        "entities": {
            "unique_sources": int(events["source_entity"].nunique()),
            "unique_destinations": int(events["destination_entity"].nunique()),
            "top_source_event_counts": {
                str(k): int(v) for k, v in events["source_entity"].value_counts().head(10).items()
            },
        },
        "edge_reuse": {
            "unique_edges": len(edge_windows),
            "edges_in_multiple_windows": sum(count > 1 for count in edge_windows.values()),
            "maximum_windows_per_edge": max(edge_windows.values(), default=0),
        },
        "qa": {
            "examples": len(qa),
            "duplicate_prompt_answer_pairs": int(qa["pair_sha256"].duplicated().sum())
            if len(qa)
            else 0,
            "unique_answers": int(qa["answer"].nunique()) if len(qa) else 0,
            "largest_training_answer_count": int(answer_counts.iloc[0])
            if len(answer_counts)
            else 0,
            "top_training_answers": {str(k): int(v) for k, v in answer_counts.head(10).items()},
        },
    }
