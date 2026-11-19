from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

from cyberslm.chat import fit_chat_history
from cyberslm.demo import analyze_structured_events, deterministic_answer

OLLAMA_GENERATE = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"
DEFAULT_CHAT_MODEL = "cyberslm-35m-chat:latest"


def ollama_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result: dict[str, Any] = json.load(response)
    return result


def available_models() -> list[str]:
    response = ollama_json(OLLAMA_TAGS)
    return [str(item["name"]) for item in response.get("models", [])]


st.set_page_config(page_title="CyberSLM", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberSLM")
st.caption("A custom 35M-parameter educational cybersecurity research model")

with st.sidebar:
    st.header("Runtime")
    model_name = st.text_input("Ollama model", DEFAULT_CHAT_MODEL)
    max_tokens = st.slider("Maximum response tokens", 32, 160, 128, 8)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    st.caption("Verified context: 512 byte tokens. Oldest complete turns are dropped first.")
    try:
        models = available_models()
        if model_name in models:
            st.success(f"Ollama ready · {model_name}")
        else:
            st.error(f"Ollama is running, but {model_name} is not registered.")
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        st.error(f"Ollama health check failed: {error}")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

chat_tab, structured_tab, status_tab = st.tabs(
    ["Experimental neural chat", "Verified structured mode", "Model status"]
)

with chat_tab:
    st.warning(
        "Experimental checkpoint: the latest frozen evaluation failed quality thresholds. "
        "Outputs may be irrelevant, incomplete, or incorrect. Do not present this as reliable."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for old_user, old_assistant in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(old_user)
        with st.chat_message("assistant"):
            st.write(old_assistant)
    user_message = st.chat_input("Ask CyberSLM an educational question")
    if user_message:
        with st.chat_message("user"):
            st.write(user_message)
        try:
            prompt, retained, prompt_tokens = fit_chat_history(
                user_message,
                st.session_state.chat_history,
                max_length=512,
                response_reserve=max_tokens,
            )
            started = time.perf_counter()
            response = ollama_json(
                OLLAMA_GENERATE,
                {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_k": 1 if temperature == 0 else 40,
                        "num_predict": max_tokens,
                        "num_ctx": 512,
                    },
                },
            )
            answer = str(response.get("response", "")).split("<|end|>", 1)[0].strip()
            latency = time.perf_counter() - started
            with st.chat_message("assistant"):
                st.write(answer or "[The model returned an empty response.]")
                st.caption(
                    f"{latency:.2f}s · {prompt_tokens} prompt tokens · "
                    f"{len(st.session_state.chat_history) - len(retained)} old turns dropped"
                )
            st.session_state.chat_history = [*retained, (user_message, answer)]
        except (
            ValueError,
            OSError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            st.error(f"Generation failed: {error}")

with structured_tab:
    st.success(
        "This mode uses verified deterministic calculations, not neural generation. "
        "Its answers must be labeled as hybrid/deterministic."
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
    raw = st.text_area(
        "Fictional structured event JSON", json.dumps(default_events, indent=2), height=260
    )
    question = st.text_input("Evidence question", "Which event type is most frequent?")
    if st.button("Run deterministic analysis"):
        try:
            events = json.loads(raw)
            facts = analyze_structured_events(events)
            st.subheader("Observed statistics")
            st.json(facts)
            st.subheader("Deterministic answer")
            st.write(deterministic_answer(facts, question))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            st.error(str(error))

with status_tab:
    st.markdown(
        """
        **Neural chat status:** experimental; acceptance and unseen quality gates failed.

        **Structured presentation model:** `cyberslm-35m-structured-byte:latest`; fixed byte GGUF
        with 180/180 native/Ollama parity on its recorded task-balanced sample.

        **Safety:** anomaly scores and event counts are review signals, not proof of an attack.
        The neural model can hallucinate and is not a substitute for an analyst.
        """
    )
