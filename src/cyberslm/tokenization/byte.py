from __future__ import annotations

import hashlib


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer used by the original CyberSLM experiments."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    vocab_size = 259

    @property
    def fingerprint(self) -> str:
        specification = b"cyberslm-byte-v1:pad=0,bos=1,eos=2,byte_offset=3"
        return hashlib.sha256(specification).hexdigest()

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [value + 3 for value in text.encode("utf-8")]
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        values: list[int] = []
        for token_id in ids:
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                if skip_special_tokens:
                    continue
                raise ValueError("Special tokens cannot be decoded as UTF-8 bytes")
            if not 3 <= token_id < self.vocab_size:
                raise ValueError(f"Token id out of range: {token_id}")
            values.append(token_id - 3)
        return bytes(values).decode("utf-8", errors="replace")
