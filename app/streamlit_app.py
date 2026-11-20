# ruff: noqa: E402
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import streamlit as st

# Make the src/ package accessible on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cyberslm.chat import fit_chat_history
from cyberslm.chat_router import parse_uploaded_bytes, route_practical_message
from cyberslm.demo import analyze_structured_events, deterministic_answer
from cyberslm.rag import KnowledgeRetriever
from cyberslm.runtime_manager import (
    AllProvidersUnavailable,
    fault_tolerant_chat,
    get_qwen_manager,
    get_scratch_manager,
)

OLLAMA_GENERATE = "http://127.0.0.1:11434/api/generate"
RAG_INDEX = Path("data/rag/cyberslm_trusted_knowledge_v1")
PRACTICAL_OLLAMA_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
SCRATCH_CHAT_MODEL = "cyberslm-35m-chat:latest"


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result: dict[str, Any] = json.load(response)
    return result


@st.cache_resource
def load_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(RAG_INDEX)


def structured_prompt(facts: dict[str, Any], question: str) -> str:
    counts = ";".join(
        f"{event_type}={count}" for event_type, count in facts["event_type_counts"].items()
    )
    return (
        "Structured cybersecurity evidence; statistics are not an attack verdict.\n"
        f"counts:{counts}\n"
        f"sources:{facts['distinct_sources']}\n"
        f"destinations:{facts['distinct_destinations']}\n"
        f"directed_edges:{facts['distinct_directed_edges']}\n"
        f"question:{question}\nReturn only the answer.\nanswer:"
    )


st.set_page_config(page_title="CyberSLM", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberSLM")
st.caption("Scratch-built 35M research model + practical grounded local assistant")

mode = st.sidebar.radio(
    "Capability",
    [
        "Practical conversational assistant",
        "Original CyberSLM-35M evidence mode",
        "Experimental CyberSLM-35M chat",
    ],
)

if mode == "Practical conversational assistant":
    st.info(
        "Model identity: Qwen2.5-1.5B-Instruct Q4_K_M (Apache-2.0), running locally. "
        "CyberSLM adds trusted-source retrieval, prompting, history, evaluation, and this UI. "
        "This is not the scratch-built 35M model."
    )
    provider_label = st.sidebar.selectbox(
        "Inference runtime",
        ["Automatic: Ollama → LM Studio", "Ollama preferred", "LM Studio preferred"],
    )
    provider = {
        "Automatic: Ollama → LM Studio": "auto",
        "Ollama preferred": "ollama",
        "LM Studio preferred": "lm_studio",
    }[provider_label]
    model = PRACTICAL_OLLAMA_MODEL
    answer_style = st.sidebar.selectbox("Answer detail", ["Balanced", "Concise", "Detailed"])
    use_rag = st.sidebar.toggle("Trusted knowledge retrieval", value=True)
    evidence_file = st.sidebar.file_uploader(
        "Optional THEIA event/anomaly results (local)",
        type=["csv", "json", "jsonl", "parquet"],
        help=(
            "Attach actual event-level results for dataset-specific questions. "
            "Local data is not NIST RAG."
        ),
    )
    evidence_rows = None
    if evidence_file is not None:
        try:
            evidence_rows = parse_uploaded_bytes(evidence_file.name, evidence_file.getvalue())
            st.sidebar.success(f"Local evidence loaded: {len(evidence_rows)} rows")
        except (ValueError, UnicodeError, ImportError, OSError) as error:
            st.sidebar.error(f"Evidence file rejected: {error}")
    score_rule = st.sidebar.selectbox(
        "Anomaly score convention",
        ["Higher score = more anomalous", "Lower score = more anomalous"],
        help=(
            "Confirm this using the actual anomaly detector. E.g. scikit-learn's "
            "decision_function: lower is more abnormal."
        ),
    )
    top_k = st.sidebar.slider("Retrieved excerpts", 2, 6, 4)
    max_tokens = st.sidebar.slider("Maximum response tokens", 128, 768, 384, 64)
    temperature = st.sidebar.slider("Temperature", 0.0, 0.8, 0.2, 0.05)
    for status in get_qwen_manager().health():
        label = f"{status.provider}: {status.model}"
        if status.reachable and status.model_available:
            st.sidebar.success(f"{label} ready")
        elif status.reachable:
            st.sidebar.warning(f"{label} unavailable")
        else:
            st.sidebar.caption(f"{label} offline")
    if st.sidebar.button("Reset conversation", use_container_width=True):
        st.session_state.practical_history = []
        st.rerun()
    if "practical_history" not in st.session_state:
        st.session_state.practical_history = []
    for item in st.session_state.practical_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item["role"] == "assistant" and item.get("verified_facts") is not None:
                with st.expander("Verified uploaded-data statistics"):
                    st.json(item["verified_facts"])
                if item.get("generated_explanation"):
                    st.markdown("**Qwen explanation (not independently verified)**")
                    st.write(item["generated_explanation"])
            if item["role"] == "assistant" and item.get("sources"):
                with st.expander("Sources used"):
                    for source in item["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
    user_message = st.chat_input("Ask a cybersecurity, AI/ML, or project question")
    if user_message:
        with st.chat_message("user"):
            st.write(user_message)
        history_for_model = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.practical_history
        ]
        try:
            response = route_practical_message(
                user_message,
                history_for_model,
                model=model,
                provider=provider,
                answer_style=answer_style,
                retriever_factory=((lambda: load_retriever()) if use_rag else (lambda: None)),
                evidence_records=evidence_rows,
                higher_is_more_anomalous=score_rule.startswith("Higher"),
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                conversation_fn=fault_tolerant_chat,
            )
            sources = [s for s in response.sources if s.get("title") and s.get("url")]
            with st.chat_message("assistant"):
                st.write(response.content)
                st.caption(
                    f"{response.latency_seconds:.2f}s · {response.provider or provider_label} · "
                    f"{response.model} · "
                    f"{len(sources)} reference sources · origin: {response.response_origin}"
                )
                if response.runtime_attempts:
                    st.caption("Runtime path: " + " → ".join(response.runtime_attempts))
                if response.verified_facts is not None:
                    st.markdown("**Verified uploaded-data statistics**")
                    st.json(response.verified_facts)
                if response.generated_explanation:
                    st.markdown("**Qwen explanation (not independently verified)**")
                    st.write(response.generated_explanation)
                if sources:
                    with st.expander("Sources used", expanded=True):
                        for source in sources:
                            st.markdown(
                                f"- [{source['title']}]({source['url']}) "
                                f"(retrieval score {source['score']:.3f})"
                            )
            st.session_state.practical_history.extend(
                [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "content": response.content,
                        "sources": sources,
                        "verified_facts": response.verified_facts,
                        "generated_explanation": response.generated_explanation,
                    },
                ]
            )
        except (
            ValueError,
            OSError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            st.error(f"Generation failed: {error}")

elif mode == "Original CyberSLM-35M evidence mode":
    st.info(
        "Model identity: the original scratch-built 35,205,120-parameter CyberSLM Transformer. "
        "Its neural output and the deterministic calculation are shown separately."
    )
    st.warning(
        "Frozen-test neural exact match is 34.38%. The deterministic hybrid is 100% because it "
        "performs verified calculations; it is not neural-model accuracy."
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
    if st.button("Analyze with original CyberSLM"):
        try:
            events = json.loads(raw)
            facts = analyze_structured_events(events)
            prompt = structured_prompt(facts, question)
            neural = get_scratch_manager().infer(prompt)
            st.subheader("Observed statistics")
            st.json(facts)
            left, right = st.columns(2)
            with left:
                st.subheader("Scratch-built neural output")
                st.code(neural.content)
                st.caption(
                    f"{neural.latency_seconds:.2f}s · {neural.model} · runtime: {neural.runtime}"
                )
            with right:
                st.subheader("Verified deterministic answer")
                st.write(deterministic_answer(facts, question))
                st.caption("Calculation engine; not generated by the neural model")
        except (
            ValueError,
            TypeError,
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            AllProvidersUnavailable,
        ) as error:
            st.error(str(error))

else:
    st.error(
        "This scratch-built conversational checkpoint failed release gates: 1/108 chat exact "
        "match and 0/1,620 structured retention. It is retained for research transparency only."
    )
    max_tokens = st.sidebar.slider("Maximum response tokens", 32, 160, 128, 8)
    if st.sidebar.button("Reset experimental chat", use_container_width=True):
        st.session_state.scratch_chat_history = []
        st.rerun()
    if "scratch_chat_history" not in st.session_state:
        st.session_state.scratch_chat_history = []
    for old_user, old_assistant in st.session_state.scratch_chat_history:
        with st.chat_message("user"):
            st.write(old_user)
        with st.chat_message("assistant"):
            st.write(old_assistant)
    user_message = st.chat_input("Experimental 35M chat prompt")
    if user_message:
        try:
            prompt, retained, _ = fit_chat_history(
                user_message,
                st.session_state.scratch_chat_history,
                max_length=512,
                response_reserve=max_tokens,
            )
            response = http_json(
                OLLAMA_GENERATE,
                {
                    "model": SCRATCH_CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_k": 1,
                        "num_predict": max_tokens,
                        "num_ctx": 512,
                    },
                },
            )
            answer = str(response.get("response", "")).split("<|end|>", 1)[0].strip()
            st.session_state.scratch_chat_history = [*retained, (user_message, answer)]
            st.rerun()
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            st.error(str(error))
