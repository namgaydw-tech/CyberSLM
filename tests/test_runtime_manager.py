from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberslm.assistant import AssistantResponse
from cyberslm.native_inference import NativeGeneration
from cyberslm.runtime_manager import (
    AllProvidersUnavailable,
    ProviderConfig,
    QwenRuntimeManager,
    ScratchRuntimeManager,
)

PROVIDERS = (
    ProviderConfig("ollama", "http://ollama.invalid", "qwen-ollama"),
    ProviderConfig("lm_studio", "http://lm.invalid", "qwen-lm"),
)


def _chat(message: str, history: list[dict[str, str]], **kwargs: object) -> AssistantResponse:
    del message, history
    provider = str(kwargs["provider"])
    model = str(kwargs["model"])
    return AssistantResponse(
        content=f"reply from {provider}",
        sources=[],
        latency_seconds=0.01,
        model=model,
        provider=provider,  # type: ignore[arg-type]
    )


def test_qwen_ollama_success() -> None:
    def request(url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        assert "ollama" in url
        return {"models": [{"name": "qwen-ollama"}]}

    result = QwenRuntimeManager(PROVIDERS, chat_fn=_chat, request_fn=request).chat("hi", [])
    assert result.provider == "ollama"
    assert result.runtime_attempts[-1].startswith("ollama:success")


def test_qwen_falls_back_to_lm_studio() -> None:
    def request(url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        if "ollama" in url:
            raise OSError("simulated Ollama outage")
        return {"data": [{"id": "qwen-lm"}]}

    result = QwenRuntimeManager(PROVIDERS, chat_fn=_chat, request_fn=request).chat("hi", [])
    assert result.provider == "lm_studio"
    assert result.runtime_attempts[0].startswith("ollama:failed")
    assert result.runtime_attempts[1].startswith("lm_studio:success")


def test_qwen_both_unavailable() -> None:
    def request(url: str, **kwargs: object) -> dict[str, object]:
        del url, kwargs
        raise TimeoutError("simulated timeout")

    with pytest.raises(AllProvidersUnavailable) as caught:
        QwenRuntimeManager(PROVIDERS, chat_fn=_chat, request_fn=request).chat("hi", [])
    assert len(caught.value.attempts) == 2


def test_qwen_recovers_preferred_provider_on_next_request() -> None:
    outage = {"active": True}

    def request(url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        if "ollama" in url:
            if outage["active"]:
                raise OSError("temporary outage")
            return {"models": [{"name": "qwen-ollama"}]}
        return {"data": [{"id": "qwen-lm"}]}

    manager = QwenRuntimeManager(PROVIDERS, chat_fn=_chat, request_fn=request)
    assert manager.chat("one", []).provider == "lm_studio"
    outage["active"] = False
    assert manager.chat("two", []).provider == "ollama"


class _FakeNative:
    checkpoint_path = "fake-checkpoint.pt"

    def generate(self, prompt: str, *, max_new_tokens: int) -> NativeGeneration:
        del prompt, max_new_tokens
        return NativeGeneration("native answer", "cpu", 0.02, "abc123")


def test_scratch_ollama_success() -> None:
    def request(url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        if url.endswith("/api/tags"):
            return {"models": [{"name": "scratch"}]}
        return {"response": "ollama answer"}

    manager = ScratchRuntimeManager(
        ollama_url="http://ollama.invalid",
        ollama_model="scratch",
        native=_FakeNative(),  # type: ignore[arg-type]
        request_fn=request,
    )
    result = manager.infer("prompt")
    assert result.runtime == "ollama"
    assert result.content == "ollama answer"


def test_scratch_native_fallback() -> None:
    def request(url: str, **kwargs: object) -> dict[str, object]:
        del url, kwargs
        raise OSError("simulated Ollama outage")

    manager = ScratchRuntimeManager(
        ollama_url="http://ollama.invalid",
        native=_FakeNative(),  # type: ignore[arg-type]
        request_fn=request,
    )
    result = manager.infer("prompt")
    assert result.runtime == "native_pytorch"
    assert result.content == "native answer"
    assert result.attempts[0].status == "failed"


def test_scratch_both_unavailable() -> None:
    broken_native = SimpleNamespace(
        checkpoint_path="missing.pt",
        generate=lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    def request(url: str, **kwargs: object) -> dict[str, object]:
        del url, kwargs
        raise OSError("offline")

    manager = ScratchRuntimeManager(
        native=broken_native,  # type: ignore[arg-type]
        request_fn=request,
    )
    with pytest.raises(AllProvidersUnavailable) as caught:
        manager.infer("prompt")
    assert [item.provider for item in caught.value.attempts] == ["ollama", "native_pytorch"]
