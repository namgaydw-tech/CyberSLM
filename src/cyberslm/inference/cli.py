from __future__ import annotations

import argparse

import torch

from cyberslm.inference.generate import generate_text
from cyberslm.models.transformer import CyberSLM, ModelConfig
from cyberslm.tokenization.bpe import BPETokenizer
from cyberslm.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a CyberSLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = CyberSLM(config)
    tokenizer = BPETokenizer.load(args.tokenizer)
    load_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=config,
        expected_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
