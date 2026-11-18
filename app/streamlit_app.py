from __future__ import annotations

import json

import streamlit as st

from cyberslm.demo import analyze_structured_events, deterministic_answer

st.set_page_config(page_title="CyberSLM Coursework Demo", layout="wide")
st.title("CyberSLM evidence-grounded coursework demo")
st.warning(
    "Educational prototype only. Event counts and anomaly scores are not proof of an attack."
)
default_events = [
    {
        "event_type": "EVENT_CONNECT",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-service-1",
        "timestamp": 1.0,
    },
    {
        "event_type": "EVENT_READ",
        "source_entity": "fictional-host-a",
        "destination_entity": "fictional-file-1",
        "timestamp": 2.5,
    },
]
raw = st.text_area("Structured event JSON", json.dumps(default_events, indent=2), height=260)
question = st.text_input("Evidence-based question", "Which event type is most frequent?")
if st.button("Analyze evidence"):
    try:
        events = json.loads(raw)
        facts = analyze_structured_events(events)
        st.subheader("Observed statistics")
        st.json(facts)
        st.subheader("Answer")
        st.write(deterministic_answer(facts, question))
        st.caption(
            "This interface currently uses the deterministic correctness baseline. "
            "The 35M checkpoint is only an engineering smoke checkpoint and is not presented "
            "as a trained cybersecurity model."
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        st.error(str(error))
