from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.checkpoint import load_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ollama_generate(url: str, model: str, prompt: str, limit: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "raw": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": limit,
                "num_ctx": 512,
                "seed": 2031,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return str(result["response"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native and Ollama completions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-model", default="cyberslm-35m-structured")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    args = parser.parse_args()

    tokenizer = ByteTokenizer()
    config = load_model_config("configs/model/cyberslm_35m.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberSLM(config)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    model.to(device).eval()
    loaded: Any = json.loads(args.prompts.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = loaded["selected"] if isinstance(loaded, dict) else loaded
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = str(example["prompt"])
        reference = str(example["reference_answer"])
        limit = min(32, len(reference.encode("utf-8")) + 1)
        prompt_ids = tokenizer.encode(prompt, bos=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            inputs,
            max_new_tokens=limit,
            top_k=1,
            eos_id=tokenizer.eos_id,
        )[0, len(prompt_ids) :].tolist()
        if tokenizer.eos_id in generated:
            generated = generated[: generated.index(tokenizer.eos_id)]
        native_output = tokenizer.decode(generated).strip()
        ollama_output = ollama_generate(args.ollama_url, args.ollama_model, prompt, limit)
        native_output_ids = tokenizer.encode(native_output)
        ollama_output_ids = tokenizer.encode(ollama_output)
        first_divergence = next(
            (
                position
                for position, (native_id, ollama_id) in enumerate(
                    zip(native_output_ids, ollama_output_ids, strict=False)
                )
                if native_id != ollama_id
            ),
            None,
        )
        if first_divergence is None and len(native_output_ids) != len(ollama_output_ids):
            first_divergence = min(len(native_output_ids), len(ollama_output_ids))
        rows.append(
            {
                **example,
                "prompt": prompt,
                "native_output": native_output,
                "ollama_output": ollama_output,
                "native_exact_match": native_output == reference,
                "ollama_exact_match": ollama_output == reference,
                "native_ollama_match": native_output == ollama_output,
                "first_divergence_token_index": first_divergence,
                "num_predict": limit,
            }
        )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": state["step"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "gguf": str(args.gguf.resolve()),
        "gguf_sha256": sha256_file(args.gguf),
        "ollama_model": args.ollama_model,
        "examples": len(rows),
        "native_exact_matches": sum(row["native_exact_match"] for row in rows),
        "ollama_exact_matches": sum(row["ollama_exact_match"] for row in rows),
        "native_ollama_matches": sum(row["native_ollama_match"] for row in rows),
        "token_sequence_matches": sum(row["first_divergence_token_index"] is None for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
