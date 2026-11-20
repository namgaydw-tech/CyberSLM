"""Run independent live API vs existing practical_chat checks on the user's laptop.

Never needs GPU training, does not mutate models, and never sends files off machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def api(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare direct inference with existing assistant")
    parser.add_argument("--provider", choices=["ollama", "lm_studio"], default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--with-rag", action="store_true")
    parser.add_argument("--out", default="outputs/cyberslm_chat_diagnostic.json")
    args = parser.parse_args()
    model = args.model or (
        "qwen2.5:1.5b-instruct-q4_K_M" if args.provider == "ollama" else "cyberslm-practical"
    )
    base = "http://127.0.0.1:11434" if args.provider == "ollama" else "http://127.0.0.1:1234"
    report = {"model": model, "provider": args.provider, "checks": []}
    try:
        available = api(base + ("/api/tags" if args.provider == "ollama" else "/v1/models"))
        available_models = (
            [m.get("name") for m in available.get("models", [])]
            if args.provider == "ollama"
            else [m.get("id") for m in available.get("data", [])]
        )
        report["available_models"] = available_models
    except Exception as exc:
        print("Inference endpoint not reachable:", type(exc).__name__, str(exc)[:350])
        return 2
    if model not in available_models:
        print("Requested model is not registered or loaded:", model)
        print("Available:", available_models)
        return 2

    from cyberslm.assistant import practical_chat
    from cyberslm.chat_router import route_practical_message

    for question in ("how are u", "what is a apt attack"):
        print("\nQUESTION:", question)
        messages = [
            {"role": "system", "content": "Respond naturally and accurately to the question."},
            {"role": "user", "content": question},
        ]
        if args.provider == "ollama":
            url = base + "/api/chat"
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 140},
            }
        else:
            url = base + "/v1/chat/completions"
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 140,
            }
        for label in ("direct_api", "existing_practical_no_rag", "patched_router"):
            begin = time.perf_counter()
            try:
                if label == "direct_api":
                    result = api(url, body)
                    answer = (
                        result.get("message", {}).get("content", "")
                        if args.provider == "ollama"
                        else result["choices"][0]["message"]["content"]
                    )
                elif label == "existing_practical_no_rag":
                    result = practical_chat(
                        question,
                        [],
                        model=model,
                        provider=args.provider,
                        answer_style="Balanced",
                        retriever=None,
                        top_k=4,
                        temperature=0.2,
                        max_tokens=140,
                    )
                    answer = result.content
                else:
                    result = route_practical_message(
                        question,
                        [],
                        model=model,
                        provider=args.provider,
                        answer_style="Balanced",
                        retriever_factory=lambda: None,
                        top_k=4,
                        temperature=0.2,
                        max_tokens=140,
                    )
                    answer = result.content
                item = {
                    "question": question,
                    "path": label,
                    "reply": str(answer),
                    "seconds": round(time.perf_counter() - begin, 3),
                }
                print(label + ":", str(answer).replace("\n", " ")[:450])
            except Exception as exc:
                item = {
                    "question": question,
                    "path": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(label + ": ERROR:", item["error"][:450])
            report["checks"].append(item)

    if args.with_rag:
        from cyberslm.rag import KnowledgeRetriever

        retriever = KnowledgeRetriever(ROOT / "data" / "rag" / "cyberslm_trusted_knowledge_v1")
        question = "What is an APT attack?"
        print("\nRAG TEST:", question)
        try:
            result = practical_chat(
                question,
                [],
                model=model,
                provider=args.provider,
                answer_style="Balanced",
                retriever=retriever,
                top_k=4,
                temperature=0.2,
                max_tokens=256,
            )
            sources = [str(s.title) for s in result.sources]
            report["rag_test"] = {
                "question": question,
                "reply": str(result.content),
                "sources": sources,
            }
            print("practical_chat with RAG:", str(result.content).replace("\n", " ")[:450])
            print("Retrieved titles:", sources)
        except Exception as exc:
            report["rag_test"] = {"error": f"{type(exc).__name__}: {exc}"}
            print("RAG test error:", str(exc))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved diagnostic report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
