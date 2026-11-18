from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result: dict[str, Any] = json.load(response)
    return result


def load_first_prompt(path: Path) -> str:
    loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = loaded["selected"] if isinstance(loaded, dict) else loaded
    if not rows:
        raise ValueError("Prompt file is empty")
    return str(rows[0]["prompt"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native tokens and first-token logits with llama.cpp server"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
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
    model.eval()
    prompt = load_first_prompt(args.prompts)
    native_ids = tokenizer.encode(prompt, bos=True)
    tokenized = post_json(
        f"{args.server_url}/tokenize",
        {"content": prompt, "add_special": True, "with_pieces": True},
    )
    llama_ids = [int(item["id"]) for item in tokenized["tokens"]]

    with torch.inference_mode():
        logits = model(torch.tensor([native_ids], dtype=torch.long)).logits[0, -1].float()
        native_log_probs = torch.log_softmax(logits, dim=-1)
        values, indices = torch.topk(native_log_probs, args.top_n)
    native_top = [
        {"id": int(index), "log_probability": float(value)}
        for index, value in zip(indices, values, strict=True)
    ]
    completion = post_json(
        f"{args.server_url}/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "n_probs": args.top_n,
            "cache_prompt": False,
        },
    )
    llama_top = completion["completion_probabilities"][0]["top_logprobs"]
    native_by_id = {item["id"]: item["log_probability"] for item in native_top}
    common_differences = {
        str(item["id"]): abs(native_by_id[item["id"]] - float(item["logprob"]))
        for item in llama_top
        if item["id"] in native_by_id
    }
    report = {
        "status": (
            "PASS"
            if native_ids == llama_ids and native_top[0]["id"] == llama_top[0]["id"]
            else "FAIL"
        ),
        "checkpoint_step": state["step"],
        "native_token_count": len(native_ids),
        "llama_cpp_token_count": len(llama_ids),
        "token_ids_identical": native_ids == llama_ids,
        "first_token_argmax_identical": native_top[0]["id"] == llama_top[0]["id"],
        "native_top_log_probs": native_top,
        "llama_cpp_top_log_probs": llama_top,
        "common_top_log_prob_absolute_differences": common_differences,
        "maximum_common_absolute_difference": max(common_differences.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
