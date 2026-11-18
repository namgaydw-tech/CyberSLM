from cyberslm.demo import analyze_structured_events, deterministic_answer


def test_demo_facts_and_non_claim() -> None:
    facts = analyze_structured_events(
        [
            {
                "event_type": "EVENT_READ",
                "source_entity": "fictional-a",
                "destination_entity": "fictional-b",
                "timestamp": 1,
            },
            {
                "event_type": "EVENT_READ",
                "source_entity": "fictional-a",
                "destination_entity": "fictional-c",
                "timestamp": 3,
            },
        ]
    )
    assert facts["event_count"] == 2
    assert facts["distinct_directed_edges"] == 2
    assert "does not prove" in deterministic_answer(facts, "Was this an APT attack?")
