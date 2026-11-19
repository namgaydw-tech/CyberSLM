from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from cyberslm.rag import KnowledgeRetriever, RetrievedChunk, format_retrieval_context

Provider = Literal["ollama", "lm_studio"]


@dataclass(frozen=True)
class AssistantResponse:
    content: str
    sources: list[RetrievedChunk]
    latency_seconds: float
    model: str
    provider: Provider
    response_origin: Literal["model", "policy_guard"] = "model"


BASE_SYSTEM_PROMPT = """You are CyberSLM Conversational Assistant, a practical local assistant
powered by the pretrained Qwen2.5-1.5B-Instruct model. You are not the scratch-built
CyberSLM-35M Transformer. Give accurate, defensive, educational cybersecurity and AI/ML help.
Maintain conversation context. Distinguish observations from confirmed incidents, never invent
evidence, and state uncertainty when information is insufficient. Do not provide instructions
whose primary purpose is unauthorized access, persistence, credential theft, evasion, or harm.
For project ideas, use fictional, local, or explicitly authorized environments. For malware and
ransomware, distinguish prevention, detection, containment, recovery, and limitations; do not imply
that paying guarantees recovery. For anomaly detection, mention false positives and human review.
Answer the current turn directly without repeating the previous answer. For project suggestions,
explicitly require synthetic or authorized data. For implementation requests, give concrete scoped
steps, suitable Python libraries, evaluation criteria, and safety boundaries. Never state that
encrypted files are recoverable only by paying: offline backups or available decryptors may help,
and payment does not guarantee recovery. Isolation Forest isolates unusual samples in fewer random
partitions; it does not learn labeled attacks, and its alerts require threshold tuning and review.
The original CyberSLM-35M is a separate 35.2M-parameter Transformer trained from random
initialization with a byte tokenizer and cybersecurity data. Its best structured neural benchmark
was 34.38%; it is not a reliable conversational model. Lead with the direct answer and address every
explicit part of the question before adding background."""


def is_disallowed_request(message: str) -> bool:
    normalized = message.lower()
    harmful_actions = (
        "steal password",
        "steal browser",
        "steal credential",
        "credential theft",
        "deploy ransomware",
        "install ransomware",
    )
    evasion = ("avoid detection", "evade detection", "remain undetected", "bypass antivirus")
    return any(item in normalized for item in harmful_actions) or (
        "password" in normalized and any(item in normalized for item in evasion)
    )


def should_retrieve(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    greetings = {"hi", "hello", "hey", "hi cyberslm", "hello cyberslm", "how are you"}
    stripped = normalized.strip(".!?")
    if "project" in stripped or stripped.startswith(
        ("summarize", "give me a python", "suggest a project", "explain how to implement")
    ):
        return False
    if stripped.startswith(("hello", "hi ", "are you")):
        return False
    return stripped not in greetings and len(stripped.split()) >= 3


def retrieval_query(message: str, history: list[dict[str, str]]) -> str:
    prior_users = [item["content"] for item in history if item["role"] == "user"][-2:]
    query = "\n".join([*prior_users, message])
    lowered = query.lower()
    if "ransomware" in lowered:
        query += "\nransomware prevention detection incident response containment backup recovery"
    if "indicator" in lowered and "compromise" in lowered:
        query += "\nindicators of compromise incident investigation false positives"
    if "mitre" in lowered or "att&ck" in lowered or "attack framework" in lowered:
        query += "\nMITRE ATT&CK Enterprise framework tactics techniques overview"
    return query


def _answer_instruction(style: str) -> str:
    if style == "Concise":
        return "Answer concisely in about 2-4 sentences unless steps are necessary."
    if style == "Detailed":
        return "Give a detailed, well-structured explanation with practical defensive examples."
    return "Use moderate detail and clear structure."


def _turn_instruction(message: str, history: list[dict[str, str]]) -> str:
    contextual = " ".join(
        [item["content"] for item in history[-4:] if item["role"] == "user"] + [message]
    ).lower()
    requirements: list[str] = []
    if "project" in contextual:
        requirements.append(
            "The first sentence must say to use only synthetic or authorized data. Recommend safe "
            "defensive work, not handling live malware."
        )
    if "implement" in message.lower():
        requirements.append(
            "Do not repeat the project title, problem statement, or description. Begin directly "
            "with a file layout or minimal code-oriented build sequence. Name concrete Python "
            "libraries, held-out evaluation, and safe local testing."
        )
    if "ransomware" in contextual and any(
        term in message.lower() for term in ("detect", "machine learning", "help")
    ):
        requirements.append(
            "Mention false positives, human review, and that this is one signal rather than proof."
        )
    if "ransomware" in message.lower() and "detect" not in message.lower():
        requirements.append(
            "Start with a compact lifecycle summary covering prevention, detection, containment, "
            "incident response, offline backups, and recovery. State that paying does not "
            "guarantee recovery before adding examples. Use at most seven concise sections so the "
            "answer finishes within the response limit."
        )
    if "isolation forest" in message.lower():
        requirements.append(
            "Explain that anomalies need fewer random partitions, then cover baseline quality, "
            "threshold tuning, false positives, concept drift, and human review."
        )
    if "cyberslm-35m" in message.lower() and "assistant" in message.lower():
        requirements.append(
            "Contrast the scratch-built 35.2M byte-token model with the current pretrained "
            "Qwen2.5-1.5B-Instruct model, and state that CyberSLM adds retrieval and local routing."
        )
    return " ".join(requirements)


def _post_json(url: str, payload: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.load(response)
    return result


def practical_chat(
    message: str,
    history: list[dict[str, str]],
    *,
    model: str = "qwen2.5:1.5b-instruct-q4_K_M",
    provider: Provider = "ollama",
    answer_style: str = "Balanced",
    retriever: KnowledgeRetriever | None = None,
    top_k: int = 4,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> AssistantResponse:
    if is_disallowed_request(message):
        return AssistantResponse(
            content=(
                "I can't help steal credentials or evade detection. I can help with defensive, "
                "authorized alternatives, for example: auditing browser credential protections, "
                "detecting credential-theft indicators, or building a safe lab exercise with "
                "synthetic data."
            ),
            sources=[],
            latency_seconds=0.0,
            model="deterministic-safety-policy",
            provider=provider,
            response_origin="policy_guard",
        )
    sources: list[RetrievedChunk] = []
    if retriever is not None and should_retrieve(message):
        query = retrieval_query(message, history)
        sources = retriever.retrieve(query, top_k=top_k)
    system = f"{BASE_SYSTEM_PROMPT}\n\n{_answer_instruction(answer_style)}"
    turn_instruction = _turn_instruction(message, history)
    if turn_instruction:
        system += f"\n\nCurrent-turn requirements: {turn_instruction}"
    if sources:
        system += (
            "\n\nUse the retrieved excerpts below when relevant. Cite only these sources using "
            "[S1], [S2], and so on. Do not invent citations. If the excerpts do not support a "
            "claim, say so.\n\n" + format_retrieval_context(sources)
        )
    messages = [{"role": "system", "content": system}, *history[-12:]]
    request_content = message
    if turn_instruction:
        request_content += f"\n\nAnswer requirements (must follow): {turn_instruction}"
    messages.append({"role": "user", "content": request_content})
    started = time.perf_counter()
    if provider == "ollama":
        result = _post_json(
            "http://127.0.0.1:11434/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 8192,
                },
            },
        )
        content = str(result["message"]["content"]).strip()
    else:
        result = _post_json(
            "http://127.0.0.1:1234/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        content = str(result["choices"][0]["message"]["content"]).strip()
    return AssistantResponse(
        content=content,
        sources=sources,
        latency_seconds=time.perf_counter() - started,
        model=model,
        provider=provider,
    )
