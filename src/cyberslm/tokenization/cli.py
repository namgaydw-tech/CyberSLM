from __future__ import annotations

import argparse

from cyberslm.data.loader import ParquetCorpus
from cyberslm.data.schema import DatasetSchema
from cyberslm.tokenization.bpe import BPETokenizer, TokenizerConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on a Parquet training corpus"
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    corpus = ParquetCorpus(args.paths, DatasetSchema(text_column=args.text_column))
    tokenizer = BPETokenizer.train(
        corpus.iter_text(),
        TokenizerConfig(vocab_size=args.vocab_size, min_frequency=args.min_frequency),
    )
    tokenizer.save(args.output)
    print(f"saved tokenizer with {tokenizer.vocab_size} tokens to {args.output}")


if __name__ == "__main__":
    main()
