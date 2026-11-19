from __future__ import annotations

from collections.abc import Sequence

from cyberslm.tokenization.byte import ByteTokenizer

USER = "<|user|>\n"
ASSISTANT = "<|assistant|>\n"
END = "\n<|end|>"


def format_chat_prompt(user: str, history: Sequence[tuple[str, str]] = ()) -> str:
    parts: list[str] = []
    for old_user, old_assistant in history:
        parts.extend((USER, old_user.strip(), "\n", ASSISTANT, old_assistant.strip(), END, "\n"))
    parts.extend((USER, user.strip(), "\n", ASSISTANT))
    return "".join(parts)


def clean_chat_completion(text: str) -> str:
    text = text.split(END, 1)[0]
    text = text.split("<|end|>", 1)[0]
    return text.strip()


def fit_chat_history(
    user: str,
    history: Sequence[tuple[str, str]],
    *,
    tokenizer: ByteTokenizer | None = None,
    max_length: int = 512,
    response_reserve: int = 128,
) -> tuple[str, list[tuple[str, str]], int]:
    tokenizer = tokenizer or ByteTokenizer()
    retained = list(history)
    while True:
        prompt = format_chat_prompt(user, retained)
        token_count = len(tokenizer.encode(prompt, bos=True))
        if token_count + response_reserve <= max_length:
            return prompt, retained, token_count
        if not retained:
            raise ValueError("Current message is too long for the configured context window")
        retained.pop(0)
