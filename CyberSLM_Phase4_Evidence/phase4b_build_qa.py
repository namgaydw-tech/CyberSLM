"""Phase 4B: construct rule-based evidence-conditioned QA review candidates.

Never uses Qwen-generated narratives as answer targets. Structured aggregate facts
are NOT independently verified source events. Does not change Phase 3 splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import pandas as pd

DEFAULT_ROOT = Path(r"D:\Design thinking Cybersecurity SLM")
SPLIT_NAMES = ("train", "validation", "cross_scenario_test", "quarantine")

def fmt_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int) and v >= 0:
        return v
    if isinstance(v, float) and v.is_integer() and v >= 0:
        return int(v)
    return None

def make_examples(row, split):
    try:
        facts = json.loads(row["verified_facts_json"])
    except (ValueError, TypeError):
        return [], "INVALID_FACTS_JSON"
    if not isinstance(facts, dict):
        return [], "NOT_FACTS_OBJECT"
    total = fmt_int(facts.get("total_events"))
    counts = facts.get("event_type_counts")
    if not isinstance(counts, dict):
        counts = {}
    count_values = {k:fmt_int(v) for k,v in counts.items() if isinstance(k,str)}
    if any(v is None for v in count_values.values()):
        return [], "BAD_EVENT_COUNTS"
    # Hard gate: an aggregate fact contradiction invalidates these targets.
    if total is not None and count_values and sum(count_values.values()) != total:
        return [], "EVENT_COUNT_INCONSISTENT"
    sources = fmt_int(facts.get("unique_sources"))
    destinations = fmt_int(facts.get("unique_destinations"))
    # Avoid too much repetitious template content; only truthful aggregate values.
    sorted_counts = sorted(count_values.items(), key=lambda x:(-x[1],x[0]))
    count_string = ", ".join(f"{k}={v}" for k,v in sorted_counts[:4])
    parts = []
    if total is not None: parts.append(f"total_events={total}")
    if count_string: parts.append("event_type_counts: " + count_string)
    if sources is not None: parts.append(f"unique_sources={sources}")
    if destinations is not None: parts.append(f"unique_destinations={destinations}")
    if not parts:
        return [], "NO_USABLE_AGGREGATE_FACTS"
    evidence = "; ".join(parts)
    base = f"Reported aggregates (unverified): {evidence}.\nQuestion: "
    pairs = []
    if total is not None:
        pairs.append(("total_events", "How many events are reported?",f"The supplied summary reports {total} events."))
    if sorted_counts:
        kind,n = sorted_counts[0]
        pairs.append(("most_common_event_type","Which event type has the highest reported count?",f"The largest reported event-type count is {kind} with {n} events."))
    if sources is not None and destinations is not None:
        pairs.append(("entity_counts","How many distinct source and destination entities are reported?",f"The summary reports {sources} distinct sources and {destinations} distinct destinations."))
    results = []
    for task, question, answer in pairs:
        prompt = base + question + "\nAnswer: "
        digest = hashlib.sha256((prompt + answer).encode("utf-8")).hexdigest()
        results.append({
            "split":split,"source_file":str(row["source_file"]),
            "source_revision":str(row.get("source_revision","")),
            "investigation_family":str(row.get("investigation_family","")),
            "qa_type":task,"prompt":prompt,"answer":answer,
            "qa_sha256":digest,"review_status":"PENDING_HUMAN_REVIEW",
            "source_evidence_status":"AGGREGATE_NOT_INDEPENDENTLY_VERIFIED",
            "training_approved":False,
        })
    return results,"OK"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output",type=Path,default=None)
    args = p.parse_args()
    split_dir = args.root / "data" / "processed" / "ismail_apt_full" / "phase3_splits"
    out = args.output or (args.root / "CyberSLM_Phase4_Evidence" / "runs" / "phase4b_qa")
    out.mkdir(parents=True,exist_ok=True)
    raw_counts={}
    rejected=Counter()
    all_frames={}
    input_digests={}
    for split in SPLIT_NAMES:
        inp=split_dir/f"{split}.parquet"
        if not inp.is_file():
            raise FileNotFoundError(inp)
        input_digests[split] = hashlib.sha256(inp.read_bytes()).hexdigest()
        df=pd.read_parquet(inp)
        raw_counts[split]=len(df)
        assert df["source_file"].is_unique, f"Duplicate source files in {split}"
        rows=[]
        for _,row in df.iterrows():
            generated,status=make_examples(row,split)
            rejected[f"{split}:{status}"]+=1
            rows.extend(generated)
        all_frames[split]=pd.DataFrame(rows)
        if all_frames[split].empty:
            raise RuntimeError(f"No QA examples for {split}; inspect structured facts")
    # Deduplicate exact prompt+answer pairs; retain original source provenance.
    existing=set()
    counts={}
    duplicates={}
    # Train-first ordering: remove duplicate examples from val/test when exact examples already in train.
    for split in SPLIT_NAMES:
        part=all_frames[split]
        before=len(part)
        part=part[~part["qa_sha256"].isin(existing)].drop_duplicates("qa_sha256").copy()
        existing.update(part["qa_sha256"].tolist())
        part=part.reset_index(drop=True)
        duplicates[split]=before-len(part)
        counts[split]=len(part)
        all_frames[split]=part
        part.to_parquet(out/f"{split}_qa.parquet",compression="zstd",index=False)
    if min(counts[name] for name in ("train","validation","cross_scenario_test")) == 0:
        raise RuntimeError("An evaluation split became empty after deduplication. Do not train; acquire more distinct evidence.")
    assert not (set(all_frames["train"]["qa_sha256"]) & set(all_frames["validation"]["qa_sha256"]))
    assert not (set(all_frames["train"]["qa_sha256"]) & set(all_frames["cross_scenario_test"]["qa_sha256"]))
    manifest={"experiment":"phase4_evidence_qa_pilot","source_files_sha256":input_digests,
              "raw_report_counts":raw_counts,"qa_counts":counts,"removed_duplicate_qa":duplicates,
              "processing_outcomes":dict(rejected),
              "answer_source":"DETERMINISTIC_RULES_FROM_UNVERIFIED_AGGREGATES",
              "human_review":"NOT COMPLETED","training_approved":False,
              "limitations":["Some source Qwen-derived aggregate facts may be incorrect.",
                             "Examples are template-generated; accuracy on them does not prove general reasoning.",
                             "The same event entities may recur between training and evaluation.",
                             "Source-event UUID lineage is not yet established.",
                             "Model should not be deployed for APT decisions."]}
    (out/"phase4b_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("PHASE 4B EVIDENCE QA REVIEW DATASET")
    print("Raw records:",raw_counts)
    print("QA examples:",counts)
    print("Exact QA duplicates removed:",duplicates)
    print("Rejection counts:",dict(rejected))
    print("Output:",out)
    print("ALL EXAMPLES PENDING HUMAN REVIEW — NOT VERIFIED GROUND TRUTH")

if __name__ == "__main__":
    main()
