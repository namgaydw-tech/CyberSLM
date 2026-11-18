import random

from scripts.build_structured_qa import TASKS, canonical_answer, make_facts
from scripts.evaluate_structured_qa import error_class, evidence_and_question, numeric_value


def test_canonical_structured_answers() -> None:
    facts = {
        "counts": {"EVENT_CONNECT": 7, "EVENT_READ": 3},
        "sources": 2,
        "destinations": 4,
        "directed_edges": 8,
        "first_type": "EVENT_CONNECT",
        "last_type": "EVENT_READ",
    }
    targets = {
        "event_count": "EVENT_READ",
        "most_frequent_event_type": None,
        "least_frequent_event_type": None,
        "event_percentage": "EVENT_READ",
        "distinct_event_types": None,
        "source_destination_counts": None,
        "directed_edge_count": None,
        "temporal_comparison": None,
        "unsupported_attack_verdict": None,
    }
    expected = ["3", "EVENT_CONNECT", "EVENT_READ", "30.0%", "2", "2,4", "8"]
    expected.extend(["EVENT_CONNECT,EVENT_READ", "NO"])
    assert [canonical_answer(task, facts, targets[task]) for task in TASKS] == expected


def test_generated_facts_are_diverse_and_unambiguous() -> None:
    facts = make_facts(random.Random(7), set())
    counts = list(facts["counts"].values())
    assert 2 <= len(counts) <= 5
    assert len(counts) == len(set(counts))


def test_numeric_parsing_and_error_classes() -> None:
    assert numeric_value("event_count", "7") == 7
    assert numeric_value("event_percentage", "30.0%") == 30.0
    assert numeric_value("source_destination_counts", "2, 4") == (2, 4)
    assert numeric_value("event_count", "EVENT_READ") is None
    assert error_class("event_count", "6", "7") == "arithmetic_error"
    assert error_class("event_count", "EVENT_READ", "7") == "parsing_failure"
    assert (
        error_class("most_frequent_event_type", "EVENT_READ", "EVENT_CONNECT")
        == "wrong_task_selection"
    )
    assert error_class("temporal_comparison", "made up", "A,B") == "hallucination"


def test_evidence_question_split() -> None:
    evidence, question = evidence_and_question("counts:A=1\nquestion:count A\nanswer:")
    assert evidence == "counts:A=1"
    assert question == "count A"
