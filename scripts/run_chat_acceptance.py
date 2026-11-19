from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from cyberslm.chat import clean_chat_completion, fit_chat_history
from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint

PROMPTS = [
    "Hi CyberSLM.",
    "What can you help me with?",
    "Explain machine learning in simple terms.",
    "What is a cybersecurity anomaly?",
    "Suggest three beginner cybersecurity projects.",
    "Explain your first suggestion.",
    "What does RoPE do in a Transformer?",
    "Can 15 network events confirm an attack?",
    "How are you different from a general LLM?",
    "Can you summarize what we discussed?",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed CyberSLM chat acceptance dialogue")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = ByteTokenizer()
    config = load_model_config("configs/model/cyberslm_35m.yaml")
    model = CyberSLM(config)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    history: list[tuple[str, str]] = []
    turns = []
    for index, user in enumerate(PROMPTS, start=1):
        prompt, retained, prompt_tokens = fit_chat_history(
            user,
            history,
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            response_reserve=args.max_new_tokens,
        )
        prompt_ids = tokenizer.encode(prompt, bos=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        started = time.perf_counter()
        generated = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            top_k=1,
            eos_id=tokenizer.eos_id,
        )[0, len(prompt_ids) :].tolist()
        eos_terminated = tokenizer.eos_id in generated
        if eos_terminated:
            generated = generated[: generated.index(tokenizer.eos_id)]
        raw = tokenizer.decode(generated)
        assistant = clean_chat_completion(raw)
        dropped = len(history) - len(retained)
        history = [*retained, (user, assistant)]
        turns.append(
            {
                "turn": index,
                "user": user,
                "assistant": assistant,
                "raw_output": raw,
                "prompt_tokens": prompt_tokens,
                "retained_prior_turns": len(retained),
                "dropped_prior_turns": dropped,
                "terminated": eos_terminated or "<|end|>" in raw,
                "latency_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(turns[-1]), flush=True)
    report = {
        "status": "COMPLETED",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": state["step"],
        "conversation_format": "<|user|>\\n...\\n<|assistant|>\\n...\\n<|end|>",
        "context_policy": "drop oldest complete turns to reserve response tokens",
        "greedy_decoding": True,
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
