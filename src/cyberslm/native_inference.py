from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint

DEFAULT_CHECKPOINT = Path(
    "outputs/cyberslm_35m_stage_b_structured_v1/checkpoints/final_step_000800.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "72185b281de6baaf80003d8bdf5cbd4e2af9602f596a01b4308ef978f314f675"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeGeneration:
    content: str
    device: str
    latency_seconds: float
    checkpoint_sha256: str


class NativeCyberSLM:
    """Lazy, checkpoint-verified native adapter for the original byte model."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
        device: str | None = None,
    ) -> None:
        configured = checkpoint_path or os.environ.get("CYBERSLM_CHECKPOINT_PATH")
        self.checkpoint_path = Path(configured) if configured else DEFAULT_CHECKPOINT
        self.expected_sha256 = expected_sha256
        self.requested_device = device
        self._model: CyberSLM | None = None
        self._tokenizer = ByteTokenizer()
        self._device: torch.device | None = None
        self._checkpoint_sha256: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return str(self._device) if self._device is not None else None

    def _select_device(self) -> torch.device:
        if self.requested_device:
            return torch.device(self.requested_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"CyberSLM checkpoint not found: {self.checkpoint_path}")
            digest = sha256_file(self.checkpoint_path)
            if self.expected_sha256 and digest != self.expected_sha256:
                raise ValueError(
                    "CyberSLM checkpoint identity mismatch: "
                    f"expected {self.expected_sha256}, observed {digest}"
                )
            config = load_model_config("configs/model/cyberslm_35m.yaml")
            model = CyberSLM(config)
            load_checkpoint(
                self.checkpoint_path,
                model=model,
                expected_model_config=config,
                expected_tokenizer_fingerprint=self._tokenizer.fingerprint,
                map_location="cpu",
            )
            device = self._select_device()
            model.to(device).eval()
            self._model = model
            self._device = device
            self._checkpoint_sha256 = digest

    def generate(self, prompt: str, *, max_new_tokens: int = 32) -> NativeGeneration:
        self.load()
        assert self._model is not None
        assert self._device is not None
        assert self._checkpoint_sha256 is not None
        prompt_ids = self._tokenizer.encode(prompt, bos=True)
        if len(prompt_ids) >= self._model.config.max_seq_len:
            raise ValueError("Prompt exceeds the native CyberSLM byte-token context")
        allowed = min(max_new_tokens, self._model.config.max_seq_len - len(prompt_ids))
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        started = time.perf_counter()
        generated = self._model.generate(
            inputs,
            max_new_tokens=allowed,
            temperature=1.0,
            top_k=1,
            eos_id=self._tokenizer.eos_id,
        )[0, len(prompt_ids) :].tolist()
        if self._tokenizer.eos_id in generated:
            generated = generated[: generated.index(self._tokenizer.eos_id)]
        return NativeGeneration(
            content=self._tokenizer.decode(generated).strip(),
            device=str(self._device),
            latency_seconds=time.perf_counter() - started,
            checkpoint_sha256=self._checkpoint_sha256,
        )
