"""Phase 4A: read-only source event schema/lineage probe.

Find small source-event Parquet candidates in the duplicated Hugging Face dataset,
inspect schema, and try a transparent edge-to-event-UUID trace. No verdict that
train/test events are independent unless explicit source-event mappings exist.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_ROOT = Path(r"D:\Design thinking Cybersecurity SLM")
DEFAULT_REPO = "namgaydw-ai/APT-Detection-Qwen-IsolationForest"
CANDIDATES = [
    "datasets_train_data/processed/theia_matched_graph_events.parquet",
    "datasets_train_data/processed/theia_unique_events.parquet",
    "datasets_train_data/processed/theia_final_integrated.parquet",
    "datasets_train_data/processed/theia_standardized.parquet",
]
ALIASES = {
    "event_id": ("event_uuid", "event_id", "edge_uuid", "record_uuid"),
    "source": ("subject_uuid", "source_entity", "source_uuid", "source", "src_uuid"),
    "destination": ("object_uuid", "destination_entity", "destination_uuid", "destination", "dst_uuid"),
    "timestamp": ("event_timestamp", "event_time_seconds", "event_time", "timestamp", "timestamp_seconds"),
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def column_match(columns, names):
    lookup = {name.lower(): name for name in columns}
    return next((lookup[x] for x in names if x in lookup), None)

def get_edges(facts_json):
    try:
        facts = json.loads(facts_json)
    except (ValueError, TypeError):
        return set()
    result = set()
    for r in facts.get("entity_relationships", []) or []:
        if isinstance(r, dict) and isinstance(r.get("source"), str):
            dest = next((r.get(k) for k in ("destination", "target", "object") if isinstance(r.get(k), str) and r.get(k)), None)
            if dest:
                result.add((r["source"].lower().strip(), dest.lower().strip()))
    return result

def trace_event_coverage(path: Path, splits: Path, cols: dict):
    # Bounded source only: do not load the large 511MB raw event file.
    usecols = [cols[x] for x in ("event_id", "source", "destination", "timestamp")]
    table = pq.read_table(path, columns=usecols).to_pandas()
    table = table.dropna(subset=usecols)
    table["_src"] = table[cols["source"]].astype(str).str.strip().str.lower()
    table["_dst"] = table[cols["destination"]].astype(str).str.strip().str.lower()
    table["_event"] = table[cols["event_id"]].astype(str).str.strip().str.lower()
    table["_time"] = pd.to_numeric(table[cols["timestamp"]], errors="coerce")
    table = table[table["_time"].notna() & table["_event"].ne("")]
    edge2ids = defaultdict(set)
    for src, dst, eid in table[["_src", "_dst", "_event"]].itertuples(index=False, name=None):
        edge2ids[(src, dst)].add(eid)
    split_results = {}
    ids_by_split = {}
    for name in ("train", "validation", "cross_scenario_test"):
        frame = pd.read_parquet(splits / f"{name}.parquet", columns=["source_file", "verified_facts_json"])
        matched_ids = set()
        edge_total = edge_hits = report_hits = 0
        for _, row in frame.iterrows():
            edges = get_edges(row["verified_facts_json"])
            edge_total += len(edges)
            hits = [edge2ids[edge] for edge in edges if edge in edge2ids]
            edge_hits += len(hits)
            report_hits += int(bool(hits))
            for ids in hits:
                matched_ids.update(ids)
        ids_by_split[name] = matched_ids
        split_results[name] = {
            "reports": len(frame), "report_edges_total": edge_total,
            "report_edges_matched_to_source": edge_hits,
            "edge_match_fraction": round(edge_hits / max(1, edge_total), 6),
            "reports_with_a_source_edge_match": report_hits,
            "distinct_source_event_ids_from_matched_edges": len(matched_ids),
        }
    overlap = {}
    for a,b in (("train","validation"),("train","cross_scenario_test"),("validation","cross_scenario_test")):
        overlap[f"{a}__{b}"] = len(ids_by_split[a] & ids_by_split[b])
    return {
        "source_rows_with_nonempty_event_uuid_and_time": len(table),
        "unique_source_event_uuids": table["_event"].nunique(),
        "source_event_uuid_conflicting_edges": int((table.groupby("_event")[["_src","_dst"]].nunique().max(axis=1)>1).sum()),
        "split_coverage": split_results,
        "matching_source_event_id_intersections": overlap,
        "interpretation": (
            "PARTIAL EVENT-LINEAGE PROBE ONLY. Zero intersection is informative ONLY if "
            "both splits have substantial exact-edge source coverage and source UUID semantics "
            "are confirmed. Shared source edge may represent several events or windows."
        ),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-mb", type=float, default=30.0)
    p.add_argument("--inspect-only", action="store_true", help="Only inspect schemas, do not map events")
    args = p.parse_args()
    out = args.output or (args.root / "CyberSLM_Phase4_Evidence" / "runs" / "phase4a_provenance")
    out.mkdir(parents=True, exist_ok=True)
    splits = args.root / "data" / "processed" / "ismail_apt_full" / "phase3_splits"
    for name in ("train", "validation", "cross_scenario_test"):
        if not (splits / f"{name}.parquet").is_file():
            raise FileNotFoundError(f"Missing frozen split: {splits / (name + '.parquet')}")
    api = HfApi()
    sha = api.repo_info(repo_id=args.repo, repo_type="dataset", token=True).sha
    tree = list(api.list_repo_tree(repo_id=args.repo, repo_type="dataset", recursive=True, revision=sha, token=True))
    indexed = {x.path:x for x in tree if hasattr(x, "size")}
    audit = {"source_repo": args.repo,"source_revision": sha,"candidates":[], "selected_source":None, "lineage_status":"UNRESOLVED"}
    local = out / "source_event_files"
    local.mkdir(exist_ok=True)
    for name in CANDIDATES:
        if name not in indexed:
            audit["candidates"].append({"file":name,"status":"NOT_PRESENT"})
            continue
        size = (indexed[name].size or 0)
        if size > args.max_mb * 1024**2:
            audit["candidates"].append({"file":name,"status":"OVER_SIZE_LIMIT","size_bytes":size})
            continue
        path = Path(hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=name, revision=sha, token=True, local_dir=str(local)))
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names
        matched = {key: column_match(columns, aliases) for key, aliases in ALIASES.items()}
        candidate = {"file":name,"status":"INSPECTED","sha256":sha256_file(path),"num_rows":pf.metadata.num_rows, "columns":columns,"matched_columns":matched}
        audit["candidates"].append(candidate)
        if args.inspect_only:
            continue
        if all(matched.values()) and audit["selected_source"] is None:
            audit["selected_source"] = name
            audit["event_trace"] = trace_event_coverage(path, splits, matched)
            audit["lineage_status"] = "PARTIAL_EDGE_TO_SOURCE_EVENT_TRACE"  # not verified independent groups
    dest = out / "phase4a_source_event_audit.json"
    dest.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print("PHASE 4A SOURCE PROVENANCE PROBE")
    print("Source revision:",sha)
    for c in audit["candidates"]:
        print(c["status"],c["file"],"rows:",c.get("num_rows"),"mapped:",c.get("matched_columns"))
    print("Trace status:",audit["lineage_status"])
    if "event_trace" in audit:
        print(json.dumps(audit["event_trace"], indent=2))
    else:
        print("No reliable event UUID + source/destination + timestamp mapping found in allowed small source files.")
    print("Saved:",dest)

if __name__ == "__main__":
    main()
