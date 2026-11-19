import pytest

from cyberslm.chat import clean_chat_completion, fit_chat_history, format_chat_prompt


def test_format_chat_prompt_matches_training_delimiters() -> None:
    prompt = format_chat_prompt("Next?", [("Hello", "Hi there")])
    assert prompt == (
        "<|user|>\nHello\n<|assistant|>\nHi there\n<|end|>\n<|user|>\nNext?\n<|assistant|>\n"
    )


def test_clean_chat_completion_stops_at_end_delimiter() -> None:
    assert clean_chat_completion("Answer.\n<|end|>ignored") == "Answer."


def test_fit_chat_history_drops_oldest_complete_turns() -> None:
    history = [("x" * 100, "y" * 100), ("a" * 30, "b" * 30)]
    prompt, retained, count = fit_chat_history(
        "current", history, max_length=220, response_reserve=40
    )
    assert retained == history[1:]
    assert "x" * 100 not in prompt
    assert count + 40 <= 220


def test_fit_chat_history_rejects_oversized_current_message() -> None:
    with pytest.raises(ValueError, match="too long"):
        fit_chat_history("x" * 600, [], max_length=512, response_reserve=16)
