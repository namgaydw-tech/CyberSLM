from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from torch.utils.data import DataLoader

from cyberslm.data.loader import ParquetCorpus
from cyberslm.models.transformer import CyberSLM, ModelConfig, count_parameters
from cyberslm.tokenization.bpe import BPETokenizer, chunk_token_stream
from cyberslm.training.dataset import CausalTokenDataset
from cyberslm.training.trainer import TrainConfig, Trainer
from cyberslm.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit CyberSLM causal pretraining")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    model_raw = load_yaml(args.model_config)
    model_raw["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**model_raw)
    train_raw = load_yaml(args.training_config)
    valid_names = {field.name for field in fields(TrainConfig)}
    config = TrainConfig(**{key: value for key, value in train_raw.items() if key in valid_names})
    corpus = ParquetCorpus(args.data)
    if corpus.validate_schema():
        raise ValueError(f"Dataset missing required columns: {corpus.validate_schema()}")
    dataset = CausalTokenDataset(
        chunk_token_stream(corpus.iter_text(), tokenizer, config.sequence_length)
    )
    loader = DataLoader(dataset, batch_size=config.batch_size)
    output = Path(args.output_dir)
    trainer = Trainer(
        CyberSLM(model_config),
        config,
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=output / "checkpoints",
        log_path=output / "metrics.jsonl",
    )
    if args.resume:
        trainer.resume(args.resume)
    print(f"model parameters: {count_parameters(trainer.model):,}")
    iterator = iter(loader)
    while trainer.state.step < config.max_steps:
        batches = []
        for _ in range(config.gradient_accumulation_steps):
            try:
                batches.append(next(iterator))
            except StopIteration:
                iterator = iter(loader)
                try:
                    batches.append(next(iterator))
                except StopIteration as error:
                    raise ValueError("Corpus did not produce any full token blocks") from error
        trainer.train_step(batches)
        if trainer.state.step % config.checkpoint_every == 0:
            trainer.save()
    final = trainer.save("final.pt")
    print(f"saved final checkpoint: {final}")


if __name__ == "__main__":
    main()
