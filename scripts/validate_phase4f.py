from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from cyberslm.data.phase4f import (
    SPLITS,
    _evidence_text,
    _task_candidates,
    _window_facts,
    load_events,
    sha256_file,
    summarize_diversity,
)
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.evidence import AnswerOnlyDataset, EvidenceExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 4F evidence QA v2")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def add_error(errors: list[dict[str, str]], code: str, detail: str) -> None:
    errors.append({"code": code, "detail": detail})


def main() -> None:
    args = parse_args()
    output = args.output or args.dataset_dir / "validation_report.json"
    errors: list[dict[str, str]] = []
    checks: dict[str, Any] = {}
    manifest = json.loads((args.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    events = load_events(args.source)
    indexed = events.set_index("event_uuid", drop=False)
    source_ids = set(indexed.index)
    windows = pd.read_parquet(args.dataset_dir / "event_window_manifest.parquet")
    qa = pd.concat(
        [pd.read_parquet(args.dataset_dir / f"{split}_qa.parquet") for split in SPLITS],
        ignore_index=True,
    )

    source_hash_matches = sha256_file(args.source) == manifest["source"]["sha256"]
    checks["source_hash_matches"] = source_hash_matches
    if not source_hash_matches:
        add_error(errors, "source_hash", "Source file differs from dataset manifest")

    all_ids: list[str] = []
    split_ids: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    window_by_id: dict[str, Any] = {}
    for window in windows.itertuples(index=False):
        ids = json.loads(window.event_uuids_json)
        all_ids.extend(ids)
        split_ids.setdefault(window.split, set()).update(ids)
        if window.split in SPLITS:
            split_groups.setdefault(window.split, set()).add(window.source_group_sha256)
        uuid_json = json.dumps(sorted(ids), separators=(",", ":"))
        expected_hash = hashlib.sha256(uuid_json.encode()).hexdigest()
        if expected_hash != window.event_uuid_sha256:
            add_error(errors, "window_hash", window.window_id)
        if set(ids) - source_ids:
            add_error(errors, "missing_uuid", window.window_id)
        window_by_id[window.window_id] = window

    checks["event_uuid_unique_across_windows"] = len(all_ids) == len(set(all_ids))
    checks["all_source_events_accounted_for"] = set(all_ids) == set(indexed.index)
    for left, right in (
        ("train", "validation"),
        ("train", "cross_scenario_test"),
        ("validation", "cross_scenario_test"),
    ):
        key = f"uuid_overlap_{left}_{right}"
        checks[key] = len(split_ids.get(left, set()) & split_ids.get(right, set()))
        group_key = f"source_group_overlap_{left}_{right}"
        checks[group_key] = len(split_groups.get(left, set()) & split_groups.get(right, set()))
        if checks[key]:
            add_error(errors, "split_uuid_overlap", key)

    checks["duplicate_qa_hashes"] = int(qa["qa_sha256"].duplicated().sum())
    checks["duplicate_prompt_answer_pairs"] = int(qa["pair_sha256"].duplicated().sum())
    checks["empty_answers"] = int(qa["answer"].fillna("").str.strip().eq("").sum())
    tokenizer = ByteTokenizer()
    recomputed = 0
    masking_checks = 0
    answer_counts: Counter[tuple[str, str]] = Counter()
    answer_limit = int(manifest["configuration"]["repeated_answer_limit"])
    max_tokens = int(manifest["configuration"]["max_sequence_tokens"])
    facts_cache: dict[str, dict[str, Any]] = {}

    for row in qa.itertuples(index=False):
        window = window_by_id.get(row.source_file)
        if window is None:
            add_error(errors, "unknown_window", row.source_file)
            continue
        if row.split != window.split or row.event_uuid_sha256 != window.event_uuid_sha256:
            add_error(errors, "qa_provenance", row.qa_sha256)
            continue
        if row.source_file not in facts_cache:
            chunk = indexed.loc[json.loads(window.event_uuids_json)]
            facts_cache[row.source_file] = _window_facts(chunk)
        facts = facts_cache[row.source_file]
        tasks = {task: (question, answer) for task, question, answer in _task_candidates(facts)}
        if row.qa_type not in tasks:
            add_error(errors, "unknown_or_redundant_task", row.qa_sha256)
            continue
        question, expected_answer = tasks[row.qa_type]
        expected_prompt = f"{_evidence_text(facts)}\nQuestion: {question}\nAnswer: "
        if row.prompt != expected_prompt or row.answer != expected_answer:
            add_error(errors, "answer_recomputation", row.qa_sha256)
            continue
        expected_pair = hashlib.sha256((row.prompt + "\0" + row.answer).encode()).hexdigest()
        if expected_pair != row.pair_sha256:
            add_error(errors, "pair_hash", row.qa_sha256)
            continue
        actual_tokens = len(tokenizer.encode(row.prompt + row.answer, bos=True, eos=True))
        if actual_tokens != row.token_count or actual_tokens > max_tokens:
            add_error(errors, "token_length", row.qa_sha256)
            continue
        dataset = AnswerOnlyDataset(
            [EvidenceExample(row.prompt, row.answer)], tokenizer=tokenizer, max_length=max_tokens
        )
        _, labels = dataset[0]
        prompt_length = len(tokenizer.encode(row.prompt))
        if not bool((labels[:prompt_length] == -100).all()) or not bool(
            (labels[prompt_length:] != -100).all()
        ):
            add_error(errors, "answer_mask", row.qa_sha256)
            continue
        masking_checks += 1
        answer_counts[(row.split, row.answer)] += 1
        recomputed += 1

    limit_violations = sum(count > answer_limit for count in answer_counts.values())
    checks["answer_limit_violations"] = limit_violations
    checks["answers_recomputed"] = recomputed
    checks["masking_examples_verified"] = masking_checks
    checks["unsupported_positive_attack_labels"] = sum(
        bool(
            ("confirmed attack" in answer.lower() or "confirmed apt attack" in answer.lower())
            and not any(term in answer.lower() for term in ("no.", "not ", "do not", "cannot"))
        )
        for answer in qa["answer"]
    )
    for field, value in checks.items():
        if isinstance(value, bool):
            if not value:
                add_error(errors, "failed_check", field)
        elif (
            isinstance(value, int)
            and field
            not in {
                "answers_recomputed",
                "masking_examples_verified",
            }
            and not field.startswith("source_group_overlap_")
            and value != 0
        ):
            add_error(errors, "failed_check", f"{field}={value}")

    report = {
        "dataset_version": manifest["dataset_version"],
        "status": "PASS" if not errors and recomputed == len(qa) else "FAIL",
        "source_rows": len(events),
        "window_rows": len(windows),
        "qa_rows": len(qa),
        "checks": checks,
        "diversity": summarize_diversity(events, windows, qa),
        "errors": errors[:100],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
