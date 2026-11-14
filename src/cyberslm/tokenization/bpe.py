from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 16_000
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")
    normalization: str = "nfkc"


@dataclass(frozen=True)
class TokenizationStats:
    text_count: int
    character_count: int
    token_count: int
    unknown_token_count: int

    @property
    def characters_per_token(self) -> float:
        return self.character_count / self.token_count if self.token_count else 0.0


class BPETokenizer:
    """Byte-level BPE wrapper with explicit special-token handling."""

    def __init__(self, tokenizer: Tokenizer, config: TokenizerConfig) -> None:
        self._tokenizer = tokenizer
        self.config = config

    @classmethod
    def train(cls, texts: Iterable[str], config: TokenizerConfig | None = None) -> BPETokenizer:
        config = config or TokenizerConfig()
        if config.vocab_size <= len(config.special_tokens):
            raise ValueError("vocab_size must exceed the number of special tokens")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        if config.normalization.lower() == "nfkc":
            tokenizer.normalizer = normalizers.NFKC()
        elif config.normalization.lower() != "none":
            raise ValueError(f"Unsupported normalization: {config.normalization}")
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=list(config.special_tokens),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer, config)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path))
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Tokenizer metadata not found: {metadata_path}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = TokenizerConfig(
            vocab_size=int(raw["vocab_size"]),
            min_frequency=int(raw["min_frequency"]),
            special_tokens=tuple(raw["special_tokens"]),
            normalization=str(raw["normalization"]),
        )
        return cls(tokenizer, config)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
        metadata = {
            "vocab_size": self.config.vocab_size,
            "min_frequency": self.config.min_frequency,
            "special_tokens": list(self.config.special_tokens),
            "normalization": self.config.normalization,
            "fingerprint": self.fingerprint,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self._tokenizer.encode_batch(list(texts))]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Unknown token: {token}")
        return token_id

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("<eos>")

    @property
    def unk_id(self) -> int:
        return self.token_to_id("<unk>")

    @property
    def fingerprint(self) -> str:
        payload = self._tokenizer.to_str().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def statistics(self, texts: Iterable[str]) -> TokenizationStats:
        count = characters = tokens = unknown = 0
        for text in texts:
            ids = self.encode(text)
            count += 1
            characters += len(text)
            tokens += len(ids)
            unknown += ids.count(self.unk_id)
        return TokenizationStats(count, characters, tokens, unknown)


def chunk_token_stream(
    texts: Iterable[str], tokenizer: BPETokenizer, sequence_length: int
) -> Iterator[list[int]]:
    """Pack documents into causal-LM blocks without loading the corpus into RAM."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    buffer: list[int] = []
    for text in texts:
        buffer.extend(tokenizer.encode(text, add_special_tokens=True))
        while len(buffer) >= sequence_length + 1:
            yield buffer[: sequence_length + 1]
            del buffer[:sequence_length]
