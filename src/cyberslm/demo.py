from __future__ import annotations

from collections import Counter
from typing import Any


def analyze_structured_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("At least one event is required")
    types = [str(event.get("event_type", "UNKNOWN")) for event in events]
    sources = {str(event["source_entity"]) for event in events if event.get("source_entity")}
    destinations = {
        str(event["destination_entity"]) for event in events if event.get("destination_entity")
    }
    edges = {
        (str(event["source_entity"]), str(event["destination_entity"]))
        for event in events
        if event.get("source_entity") and event.get("destination_entity")
    }
    times = sorted(
        float(event["timestamp"]) for event in events if event.get("timestamp") is not None
    )
    counts = Counter(types)
    top_type, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        "event_count": len(events),
        "event_type_counts": dict(sorted(counts.items())),
        "distinct_sources": len(sources),
        "distinct_destinations": len(destinations),
        "distinct_directed_edges": len(edges),
        "top_event_type": top_type,
        "top_event_type_count": top_count,
        "duration_seconds": times[-1] - times[0] if len(times) > 1 else 0.0,
        "limitations": [
            "The submitted records are treated as observations, not verified attack labels.",
            "Counts or anomaly scores do not prove malicious activity.",
        ],
    }


def deterministic_answer(facts: dict[str, Any], question: str) -> str:
    lowered = question.lower()
    if "how many" in lowered and "event" in lowered:
        for event_type, count in facts["event_type_counts"].items():
            if event_type.lower() in lowered:
                return f"There are {count} recorded {event_type} events."
        return f"There are {facts['event_count']} recorded events."
    if "most frequent" in lowered or "most common" in lowered:
        noun = "event" if facts["top_event_type_count"] == 1 else "events"
        return (
            f"{facts['top_event_type']} is most frequent with "
            f"{facts['top_event_type_count']} {noun}."
        )
    if "edge" in lowered:
        return f"There are {facts['distinct_directed_edges']} distinct directed edges."
    if "attack" in lowered or "apt" in lowered:
        return "No. The supplied event evidence alone does not prove a confirmed attack."
    return (
        "The evidence supports aggregate counts only; the question requires either a more "
        "specific calculation or independently verified context."
    )
