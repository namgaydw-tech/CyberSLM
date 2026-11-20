from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from cyberslm.assistant import AssistantResponse, practical_chat
from cyberslm.native_inference import NativeCyberSLM

ProviderName = Literal["ollama", "lm_studio"]


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    base_url: str
    model: str
    connect_timeout_seconds: float = 3.0
    request_timeout_seconds: float = 180.0


@dataclass(frozen=True)
class RuntimeAttempt:
    provider: str
    model: str
    status: str
    detail: str
    latency_seconds: float


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    base_url: str
    model: str
    reachable: bool
    model_available: bool
    detail: str


@dataclass(frozen=True)
class ScratchResponse:
    content: str
    model: str
    runtime: str
    latency_seconds: float
    attempts: tuple[RuntimeAttempt, ...]
    checkpoint_sha256: str | None = None


class AllProvidersUnavailable(RuntimeError):
    def __init__(self, service: str, attempts: list[RuntimeAttempt]) -> None:
        self.service = service
        self.attempts = tuple(attempts)
        details = "; ".join(f"{item.provider}: {item.detail}" for item in attempts)
        super().__init__(f"{service} unavailable ({details})")


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: dict[str, Any] = json.load(response)
    return result


def default_qwen_providers() -> tuple[ProviderConfig, ProviderConfig]:
    return (
        ProviderConfig(
            "ollama",
            os.environ.get("CYBERSLM_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            os.environ.get("CYBERSLM_OLLAMA_QWEN_MODEL", "qwen2.5:1.5b-instruct-q4_K_M"),
        ),
        ProviderConfig(
            "lm_studio",
            os.environ.get("CYBERSLM_LM_STUDIO_URL", "http://127.0.0.1:1234").rstrip("/"),
            os.environ.get("CYBERSLM_LM_STUDIO_MODEL", "cyberslm-practical"),
        ),
    )


class QwenRuntimeManager:
    def __init__(
        self,
        providers: tuple[ProviderConfig, ...] | None = None,
        *,
        chat_fn: Callable[..., AssistantResponse] = practical_chat,
        request_fn: Callable[..., dict[str, Any]] = _json_request,
    ) -> None:
        self.providers = providers or default_qwen_providers()
        self.chat_fn = chat_fn
        self.request_fn = request_fn

    def _models(self, provider: ProviderConfig) -> list[str]:
        path = "/api/tags" if provider.name == "ollama" else "/v1/models"
        payload = self.request_fn(
            provider.base_url + path,
            timeout=provider.connect_timeout_seconds,
        )
        if provider.name == "ollama":
            return [str(item.get("name")) for item in payload.get("models", [])]
        return [str(item.get("id")) for item in payload.get("data", [])]

    def health(self) -> list[ProviderStatus]:
        statuses: list[ProviderStatus] = []
        for provider in self.providers:
            try:
                models = self._models(provider)
                available = provider.model in models
                statuses.append(
                    ProviderStatus(
                        provider.name,
                        provider.base_url,
                        provider.model,
                        True,
                        available,
                        "ready" if available else f"model not available; loaded={models}",
                    )
                )
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
                statuses.append(
                    ProviderStatus(
                        provider.name,
                        provider.base_url,
                        provider.model,
                        False,
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        return statuses

    def chat(
        self,
        message: str,
        history: list[dict[str, str]],
        *,
        preferred: Literal["auto", "ollama", "lm_studio"] = "auto",
        **kwargs: Any,
    ) -> AssistantResponse:
        candidates = list(self.providers)
        if preferred != "auto":
            candidates.sort(key=lambda item: item.name != preferred)
        attempts: list[RuntimeAttempt] = []
        for provider in candidates:
            started = time.perf_counter()
            try:
                models = self._models(provider)
                if provider.model not in models:
                    raise LookupError(f"model {provider.model!r} is not loaded")
                result = self.chat_fn(
                    message,
                    history,
                    model=provider.model,
                    provider=provider.name,
                    request_timeout=provider.request_timeout_seconds,
                    **kwargs,
                )
                attempts.append(
                    RuntimeAttempt(
                        provider.name,
                        provider.model,
                        "success",
                        "request completed",
                        time.perf_counter() - started,
                    )
                )
                return replace(
                    result,
                    runtime_attempts=tuple(
                        f"{item.provider}:{item.status}:{item.detail}" for item in attempts
                    ),
                )
            except (
                OSError,
                TimeoutError,
                urllib.error.URLError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                LookupError,
            ) as exc:
                attempts.append(
                    RuntimeAttempt(
                        provider.name,
                        provider.model,
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                        time.perf_counter() - started,
                    )
                )
        raise AllProvidersUnavailable("Qwen explanation service", attempts)


class ScratchRuntimeManager:
    def __init__(
        self,
        *,
        ollama_url: str | None = None,
        ollama_model: str = "cyberslm-35m-structured-byte:latest",
        native: NativeCyberSLM | None = None,
        request_fn: Callable[..., dict[str, Any]] = _json_request,
    ) -> None:
        self.ollama_url = (
            ollama_url or os.environ.get("CYBERSLM_OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.ollama_model = ollama_model
        self.native = native or NativeCyberSLM()
        self.request_fn = request_fn

    def infer(self, prompt: str, *, max_new_tokens: int = 32) -> ScratchResponse:
        attempts: list[RuntimeAttempt] = []
        started = time.perf_counter()
        try:
            tags = self.request_fn(self.ollama_url + "/api/tags", timeout=3.0)
            models = [str(item.get("name")) for item in tags.get("models", [])]
            if self.ollama_model not in models:
                raise LookupError(f"model {self.ollama_model!r} is not registered")
            result = self.request_fn(
                self.ollama_url + "/api/generate",
                payload={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_k": 1,
                        "top_p": 1,
                        "min_p": 0,
                        "repeat_penalty": 1,
                        "num_ctx": 512,
                        "num_predict": max_new_tokens,
                    },
                },
                timeout=180.0,
            )
            content = str(result["response"]).strip()
            attempts.append(
                RuntimeAttempt(
                    "ollama",
                    self.ollama_model,
                    "success",
                    "request completed",
                    time.perf_counter() - started,
                )
            )
            return ScratchResponse(
                content,
                self.ollama_model,
                "ollama",
                time.perf_counter() - started,
                tuple(attempts),
            )
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            LookupError,
        ) as exc:
            attempts.append(
                RuntimeAttempt(
                    "ollama",
                    self.ollama_model,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    time.perf_counter() - started,
                )
            )
        started = time.perf_counter()
        try:
            native = self.native.generate(prompt, max_new_tokens=max_new_tokens)
            attempts.append(
                RuntimeAttempt(
                    "native_pytorch",
                    str(self.native.checkpoint_path),
                    "success",
                    f"device={native.device}",
                    native.latency_seconds,
                )
            )
            return ScratchResponse(
                native.content,
                "CyberSLM-35M checkpoint",
                "native_pytorch",
                time.perf_counter() - started,
                tuple(attempts),
                native.checkpoint_sha256,
            )
        except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
            attempts.append(
                RuntimeAttempt(
                    "native_pytorch",
                    str(self.native.checkpoint_path),
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    time.perf_counter() - started,
                )
            )
            raise AllProvidersUnavailable("CyberSLM scratch inference", attempts) from exc


_QWEN_MANAGER: QwenRuntimeManager | None = None
_SCRATCH_MANAGER: ScratchRuntimeManager | None = None


def get_qwen_manager() -> QwenRuntimeManager:
    global _QWEN_MANAGER
    if _QWEN_MANAGER is None:
        _QWEN_MANAGER = QwenRuntimeManager()
    return _QWEN_MANAGER


def get_scratch_manager() -> ScratchRuntimeManager:
    global _SCRATCH_MANAGER
    if _SCRATCH_MANAGER is None:
        _SCRATCH_MANAGER = ScratchRuntimeManager()
    return _SCRATCH_MANAGER


def fault_tolerant_chat(
    message: str,
    history: list[dict[str, str]],
    *,
    provider: str = "auto",
    model: str | None = None,
    **kwargs: Any,
) -> AssistantResponse:
    del model
    preferred = cast(
        Literal["auto", "ollama", "lm_studio"],
        provider if provider in {"ollama", "lm_studio"} else "auto",
    )
    return get_qwen_manager().chat(message, history, preferred=preferred, **kwargs)
