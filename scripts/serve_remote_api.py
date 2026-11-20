"""Authenticated CyberSLM backend contract for a private HTTPS reverse proxy.

This process intentionally serves HTTP locally. Put it behind an authorized TLS
reverse proxy or tunnel; never expose Ollama or LM Studio ports directly.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cyberslm.runtime_manager import AllProvidersUnavailable, fault_tolerant_chat
from cyberslm.two_model_bridge import run_bridge

MAX_BODY = 1_000_000
WINDOW_SECONDS = 60
MAX_REQUESTS = 30
REQUESTS: dict[str, deque[float]] = defaultdict(deque)


class Handler(BaseHTTPRequestHandler):
    server_version = "CyberSLM-API/1.0"

    def _json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-content-type-options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        configured = os.environ.get("CYBERSLM_API_TOKEN", "")
        supplied = self.headers.get("authorization", "").removeprefix("Bearer ")
        return bool(configured) and hmac.compare_digest(configured, supplied)

    def _rate_allowed(self) -> bool:
        now = time.monotonic()
        bucket = REQUESTS[self.client_address[0]]
        while bucket and bucket[0] < now - WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= MAX_REQUESTS:
            return False
        bucket.append(now)
        return True

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("content-length", "0"))
        if size <= 0 or size > MAX_BODY:
            raise ValueError("Request body must be between 1 byte and 1 MB")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"ok": False})
            return
        self._json(HTTPStatus.OK, {"ok": True, "service": "CyberSLM backend"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        if not self._rate_allowed():
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"ok": False, "error": "rate limited"})
            return
        try:
            body = self._body()
            if self.path == "/v1/chat":
                self._chat(body)
            elif self.path == "/v1/bridge":
                self._bridge(body)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except AllProvidersUnavailable as exc:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": str(exc), "attempts": [asdict(x) for x in exc.attempts]},
            )

    def _chat(self, body: dict[str, Any]) -> None:
        message = body.get("message")
        history = body.get("history", [])
        if not isinstance(message, str) or not 0 < len(message) <= 2_000:
            raise ValueError("message must be 1-2,000 characters")
        if not isinstance(history, list):
            raise ValueError("history must be a list")
        result = fault_tolerant_chat(
            message,
            history[-12:],
            provider="auto",
            retriever=None,
            max_tokens=512,
        )
        self._json(
            HTTPStatus.OK,
            {
                "ok": True,
                "mode": "conversation",
                "content": result.content,
                "provider": result.provider,
                "model": result.model,
                "attempts": result.runtime_attempts,
            },
        )

    def _bridge(self, body: dict[str, Any]) -> None:
        question = body.get("question")
        records = body.get("records")
        if not isinstance(question, str) or not 0 < len(question) <= 400:
            raise ValueError("question must be 1-400 characters")
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        result = run_bridge(records, question, provider="auto")
        self._json(HTTPStatus.OK, {"ok": True, "mode": "bridge", **asdict(result)})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not os.environ.get("CYBERSLM_API_TOKEN"):
        raise SystemExit("Set CYBERSLM_API_TOKEN to a long random secret before startup")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CyberSLM backend listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
