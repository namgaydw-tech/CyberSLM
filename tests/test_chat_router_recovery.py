"""Offline regression suite: no GPU, Ollama, LM Studio, or private THEIA data needed."""

import csv
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cyberslm.chat_router import (
    classify_question,
    parse_uploaded_bytes,
    route_practical_message,
    verified_evidence_summary,
)

ROWS = [
    {
        "event_id": "a",
        "event_type": "EVENT_READ",
        "size_bin": "S",
        "sequence_bucket": "first",
        "anomaly_score": "0.2",
    },
    {
        "event_id": "b",
        "event_type": "EVENT_READ",
        "size_bin": "S",
        "sequence_bucket": "second",
        "anomaly_score": "0.9",
    },
    {
        "event_id": "c",
        "event_type": "EVENT_WRITE",
        "size_bin": "L",
        "sequence_bucket": "second",
        "anomaly_score": "0.4",
    },
]


def conv(prompt, history, **kwargs):
    return SimpleNamespace(
        content="Qwen wrote an answer.",
        model=kwargs["model"],
        latency_seconds=0.2,
        sources=[],
        response_origin="model",
    )


class ChatRouterTest(unittest.TestCase):
    def _route(self, question, rows=None, *, history=(), conv_fn=conv, greeting_fn=None):
        return route_practical_message(
            question,
            history,
            model="qwen2.5:1.5b-instruct-q4_K_M",
            provider="ollama",
            answer_style="Balanced",
            retriever_factory=lambda: object(),
            top_k=4,
            temperature=0.2,
            max_tokens=384,
            evidence_records=rows,
            conversation_fn=conv_fn,
            greeting_fn=greeting_fn,
        )

    def test_greeting_does_not_require_evidence_or_retrieval(self):
        seen = []

        def fake_greeting(q, **kw):
            seen.append(q)
            return "Hello! How can I help?", 0.1

        response = self._route(
            "how are u",
            greeting_fn=fake_greeting,
            history=[
                {"role": "user", "content": "THEIA anomalies"},
                {"role": "assistant", "content": "I do not have access to dataset"},
            ],
        )
        self.assertEqual(response.response_origin, "local_model_greeting")
        self.assertEqual(response.content, "Hello! How can I help?")
        self.assertEqual(seen, ["how are u"])

    def test_factual_general_question_not_misclassified_as_data(self):
        self.assertEqual(classify_question("what is a apt attack"), "general")
        response = self._route("what is a apt attack")
        self.assertEqual(response.content, "Qwen wrote an answer.")

    def test_specific_data_question_without_records_invokes_qwen(self):
        for prompt in (
            "Identify the most unusual events in supplied THEIA anomaly results",
            "What are the most frequently observed event types in the available dataset?",
            "Are there unusual patterns in operation types, size bins, or sequence buckets?",
            (
                "Based on the available anomaly scores and raw-event evidence, is there "
                "sufficient evidence to suspect an APT attack? Explain what is known "
                "and what additional evidence is needed."
            ),
        ):
            with self.subTest(prompt=prompt):
                calls = []
                def spy(message, history, calls_ref=calls, **kwargs):
                    calls_ref.append((message, history, kwargs))
                    return SimpleNamespace(
                        content=(
                            "Without actual results, an assessment requires a baseline "
                            "and timeline."
                        ),
                        model=kwargs["model"],
                        latency_seconds=0.3,
                        sources=[],
                        response_origin="model",
                    )

                output = self._route(prompt, conv_fn=spy)
                self.assertEqual(output.response_origin, "model_without_evidence")
                self.assertGreater(output.latency_seconds, 0)
                self.assertIn("assessment", output.content)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], [])  # Old refusals never contaminate Qwen
                self.assertIn("No actual THEIA", calls[0][0])
                self.assertIn("Do not merely refuse", calls[0][0])

    def test_next_event_forecast_explained_not_fabricated(self):
        calls = []

        def spy(message, history, **kwargs):
            calls.append(message)
            return SimpleNamespace(
                content="A trained sequence forecaster and a feature schema are necessary.",
                model=kwargs["model"],
                latency_seconds=0.1,
                sources=[],
                response_origin="model",
            )

        resp = self._route(
            "Given the previous 32 raw events, predict the next event's eight features",
            ROWS,
            conv_fn=spy,
        )
        self.assertEqual(resp.response_origin, "forecast_explanation_unvalidated")
        self.assertIn("sequence forecaster", resp.content)
        self.assertIn("do not predict feature", calls[0])

    def test_missing_evidence_generic_refusal_gets_honest_guidance(self):
        def refusal(message, history, **kwargs):
            return SimpleNamespace(
                content=(
                    "I'm sorry but I don't have access to the anomaly results. "
                    "Please upload them."
                ),
                model=kwargs["model"],
                latency_seconds=0.1,
                sources=[],
                response_origin="model",
            )

        resp = self._route(
            (
                "Based on the available anomaly scores and raw-event evidence, is there "
                "sufficient evidence to suspect an APT attack?"
            ),
            conv_fn=refusal,
        )
        self.assertEqual(resp.response_origin, "general_guidance_fallback")
        self.assertIn("persistence", resp.content)
        self.assertIn("cannot conclude", resp.content)
        self.assertEqual(resp.sources, [])
        self.assertGreater(resp.latency_seconds, 0)

    def test_frequency_summary_uses_real_records(self):
        result = verified_evidence_summary(
            ROWS,
            "What are the most frequently observed event types in the available dataset?",
            higher_is_more_anomalous=True,
        )
        self.assertEqual(
            result["verified_statistics"]["event_type_counts"], {"EVENT_READ": 2, "EVENT_WRITE": 1}
        )
        self.assertEqual(result["verified_statistics"]["size_bin_counts"], {"S": 2, "L": 1})
        self.assertEqual(
            result["verified_statistics"]["sequence_bucket_counts"], {"second": 2, "first": 1}
        )

    def test_anomaly_ranking_requires_explicit_orientation(self):
        q = "Identify the most unusual events in supplied THEIA anomaly results"
        high = verified_evidence_summary(ROWS, q, higher_is_more_anomalous=True)
        low = verified_evidence_summary(ROWS, q, higher_is_more_anomalous=False)
        self.assertEqual(high["most_anomalous_records"][0]["event_id"], "b")
        self.assertEqual(low["most_anomalous_records"][0]["event_id"], "a")
        self.assertIn("do not reveal causal", high["message"])

    def test_no_score_column_has_no_fake_rank(self):
        result = verified_evidence_summary(
            [{"event_type": "EVENT_READ"}], "unusual events in THEIA", higher_is_more_anomalous=True
        )
        self.assertEqual(result["score_status"], "NO_NUMERIC_ANOMALY_SCORE_COLUMN")
        self.assertNotIn("most_anomalous_records", result)

    def test_safe_data_context_and_separate_explanation(self):
        prompts = []

        def spy(prompt, history, **kwargs):
            prompts.append((prompt, history, kwargs.get("retriever")))
            return conv(prompt, history, **kwargs)

        result = self._route(
            "What are the most frequently observed event types in the available dataset?",
            ROWS,
            conv_fn=spy,
        )
        self.assertEqual(result.response_origin, "verified_local_evidence_plus_model")
        self.assertIsNotNone(result.verified_facts)
        self.assertEqual(result.generated_explanation, "Qwen wrote an answer.")
        self.assertIn("EVENT_READ", prompts[0][0])
        self.assertEqual(prompts[0][1], [])
        self.assertIsNone(prompts[0][2])

    def test_history_cleanup_removes_previous_data_refusals(self):
        saved = []

        def spy(prompt, history, **kwargs):
            saved.extend(history)
            return conv(prompt, history, **kwargs)

        history = [
            {"role": "user", "content": "identify unusual THEIA anomaly scores"},
            {"role": "assistant", "content": "I do not have access to the anomaly results"},
            {"role": "user", "content": "What is a firewall?"},
            {"role": "assistant", "content": "A firewall controls network flows."},
        ]
        self._route("Explain an APT attack", history=history, conv_fn=spy)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["content"], "What is a firewall?")

    def test_csv_json_and_jsonl_readers(self):
        csvdata = b"event_id,event_type\na,EVENT_READ\nb,EVENT_WRITE\n"
        self.assertEqual(len(parse_uploaded_bytes("test.csv", csvdata)), 2)
        self.assertEqual(
            len(parse_uploaded_bytes("test.json", json.dumps({"events": ROWS}).encode())), 3
        )
        self.assertEqual(
            len(parse_uploaded_bytes("test.jsonl", b'{"event_type":"EVENT_READ"}\n')), 1
        )

    def test_csv_256_rows_are_counted_exactly(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["event_id", "event_type"])
        writer.writeheader()
        for index in range(256):
            writer.writerow({"event_id": index, "event_type": "EVENT_READ"})
        rows = parse_uploaded_bytes("synthetic-256.csv", output.getvalue().encode())
        summary = verified_evidence_summary(
            rows,
            "What are the event types in the available dataset?",
            higher_is_more_anomalous=True,
        )
        self.assertEqual(len(rows), 256)
        self.assertEqual(summary["verified_statistics"]["rows_available"], 256)
        self.assertEqual(summary["verified_statistics"]["event_type_counts"], {"EVENT_READ": 256})

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_uploaded_bytes("x.csv", b"")
        with self.assertRaises(ValueError):
            parse_uploaded_bytes("x.json", b"[1,2,3]")


if __name__ == "__main__":
    unittest.main()
