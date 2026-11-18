from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberslm.export.gguf import export_checkpoint_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CyberSLM checkpoint to experimental GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    report = export_checkpoint_to_gguf(args.checkpoint, args.output, output_dtype=args.dtype)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
