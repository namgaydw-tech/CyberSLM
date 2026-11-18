from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from cyberslm.data.phase4f import _window_facts, load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-free Phase 6 anomaly evidence")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        name: float(values.quantile(level))
        for name, level in (
            ("min", 0.0),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("max", 1.0),
        )
    }


def main() -> None:
    args = parse_args()
    events = load_events(args.source).set_index("event_uuid", drop=False)
    windows = pd.read_parquet(args.windows)
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        if window.split == "quarantine":
            continue
        chunk = events.loc[json.loads(window.event_uuids_json)]
        facts = _window_facts(chunk)
        top_count = facts["ranked"][0][1]
        rows.append(
            {
                "window_id": window.window_id,
                "split": window.split,
                "event_count": facts["total"],
                "event_type_diversity": len(facts["counts"]),
                "event_type_entropy_bits": facts["entropy"],
                "top_event_type_share": top_count / facts["total"],
                "distinct_sources": facts["sources"],
                "distinct_destinations": facts["destinations"],
                "distinct_edges": facts["distinct_edges"],
                "repeated_edge_occurrences": facts["repeated_edges"],
                "adjacent_type_transitions": facts["transitions"],
                "duration_seconds": facts["duration"],
            }
        )
    features = pd.DataFrame(rows)
    feature_names = [column for column in features.columns if column not in {"window_id", "split"}]
    train = features[features["split"] == "train"]
    if len(train) < 3:
        raise ValueError("At least three training windows are required")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train[feature_names])
    all_values = scaler.transform(features[feature_names])

    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    ).fit(train_values)
    neighbors = min(20, len(train) - 1)
    lof = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, contamination="auto").fit(
        train_values
    )
    # Higher values consistently mean more unusual.
    features["isolation_anomaly_score"] = -isolation.score_samples(all_values)
    features["lof_anomaly_score"] = -lof.score_samples(all_values)
    features["isolation_flag"] = isolation.predict(all_values) == -1
    features["lof_flag"] = lof.predict(all_values) == -1
    features["evidence_status"] = "ALGORITHMIC_SCORE_NOT_ATTACK_LABEL"
    features["interpretation"] = (
        "An unusual feature profile is a review signal, not proof of malicious activity."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "window_algorithm_evidence.parquet"
    features.to_parquet(scores_path, index=False)
    split_summary: dict[str, Any] = {}
    for split, group in features.groupby("split", sort=True):
        split_summary[split] = {
            "windows": len(group),
            "isolation_score_quantiles": quantiles(group["isolation_anomaly_score"]),
            "lof_score_quantiles": quantiles(group["lof_anomaly_score"]),
            "isolation_flags": int(group["isolation_flag"].sum()),
            "lof_flags": int(group["lof_flag"].sum()),
        }
    report = {
        "phase": 6,
        "methods": [
            "IsolationForest",
            "LocalOutlierFactor(novelty=True)",
            "directed_graph_features",
        ],
        "feature_names": feature_names,
        "training_windows": len(train),
        "split_summary": split_summary,
        "label_status": (
            "No verified attack/benign labels used; no classification accuracy reported."
        ),
        "score_direction": "Higher scores indicate greater feature-space unusualness.",
        "algorithm_evidence_counts": dict(Counter(features["evidence_status"])),
    }
    (args.output_dir / "phase6_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
