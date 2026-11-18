from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from cyberslm.models.configuration import load_model_config
from cyberslm.models.transformer import CyberSLM, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_template(path: Path, values: dict[str, str]) -> str:
    content = path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace("{{" + key + "}}", value)
    if "{{" in content or "}}" in content:
        raise ValueError("Unresolved model-card template placeholder")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Package custom CyberSLM weights for HF Hub")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/cyberslm_35m.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="CyberSLM-35M")
    parser.add_argument("--training-stage", required=True)
    parser.add_argument(
        "--artifact-status",
        choices=("bounded-trained-checkpoint", "engineering-smoke-test"),
        required=True,
    )
    parser.add_argument("--training-description", required=True)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument(
        "--gguf",
        type=Path,
        help="Optional validated GGUF export to include as an additional inference artifact",
    )
    parser.add_argument("--template", type=Path, default=Path("docs/templates/hf_model_card.md"))
    args = parser.parse_args()

    payload: dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("model"), dict):
        raise ValueError("Checkpoint lacks a model state dictionary")
    config = load_model_config(str(args.model_config))
    expected_config = vars(config)
    if payload.get("model_config") != expected_config:
        raise ValueError("Checkpoint model configuration does not match --model-config")
    tokenizer = ByteTokenizer()
    if payload.get("tokenizer_fingerprint") != tokenizer.fingerprint:
        raise ValueError("Checkpoint tokenizer fingerprint mismatch")
    model = CyberSLM(config)
    model.load_state_dict(payload["model"], strict=True)
    parameter_count = count_parameters(model)
    if parameter_count != 35_205_120:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference_path = args.output_dir / "pytorch_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": expected_config,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "training_state": payload.get("state", {}),
            "format": "cyberslm-custom-pytorch-inference-v1",
        },
        inference_path,
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "cyberslm_custom",
                "architectures": ["CyberSLM"],
                **expected_config,
                "parameter_count": parameter_count,
                "auto_model_compatible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_type": "cyberslm_utf8_byte_v1",
                "vocab_size": tokenizer.vocab_size,
                "pad_token_id": tokenizer.pad_id,
                "bos_token_id": tokenizer.bos_id,
                "eos_token_id": tokenizer.eos_id,
                "byte_offset": 3,
                "fingerprint": tokenizer.fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("src/cyberslm/models/transformer.py", args.output_dir / "modeling_cyberslm.py")
    shutil.copy2("src/cyberslm/tokenization/byte.py", args.output_dir / "tokenization_cyberslm.py")
    if args.training_summary:
        shutil.copy2(args.training_summary, args.output_dir / "training_summary.json")
    if args.gguf:
        if not args.gguf.is_file():
            raise FileNotFoundError(args.gguf)
        shutil.copy2(args.gguf, args.output_dir / args.gguf.name)
    load_script = """from __future__ import annotations

import argparse
import json

import torch

from modeling_cyberslm import CyberSLM, ModelConfig
from tokenization_cyberslm import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    metadata = json.loads(open("config.json", encoding="utf-8").read())
    fields = {key: metadata[key] for key in ModelConfig.__dataclass_fields__}
    model = CyberSLM(ModelConfig(**fields))
    payload = torch.load("pytorch_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = ByteTokenizer()
    if payload["tokenizer_fingerprint"] != tokenizer.fingerprint:
        raise ValueError("Tokenizer fingerprint mismatch")
    ids = torch.tensor([tokenizer.encode(args.prompt, bos=True)], dtype=torch.long)
    result = model.generate(ids, max_new_tokens=args.max_new_tokens, top_k=1, eos_id=2)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
"""
    (args.output_dir / "load_model.py").write_text(load_script, encoding="utf-8")
    (args.output_dir / "requirements.txt").write_text("torch>=2.5,<3\n", encoding="utf-8")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    card = render_template(
        args.template,
        {
            "MODEL_NAME": args.model_name,
            "ARTIFACT_STATUS": args.artifact_status,
            "TRAINING_STAGE": args.training_stage,
            "PARAMETER_COUNT": f"{parameter_count:,}",
            "CHECKPOINT_SHA256": sha256_file(args.checkpoint),
            "CODE_REVISION": revision,
            "TRAINING_DESCRIPTION": args.training_description,
        },
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    receipt_files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "checksums.json"
    )
    receipts = {path.name: sha256_file(path) for path in receipt_files}
    (args.output_dir / "checksums.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "checksums": receipts}, indent=2))


if __name__ == "__main__":
    main()
