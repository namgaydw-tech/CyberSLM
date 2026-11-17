from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from cyberslm.data.loader import ParquetCorpus
from cyberslm.inference.generate import generate_text
from cyberslm.models.transformer import CyberSLM, ModelConfig
from cyberslm.tokenization.bpe import BPETokenizer, TokenizerConfig, chunk_token_stream
from cyberslm.training.dataset import CausalTokenDataset
from cyberslm.training.trainer import TrainConfig, Trainer


@pytest.mark.slow
def test_end_to_end_cpu_pipeline_and_tiny_overfit(tmp_path: Path) -> None:
    corpus = ParquetCorpus("data/fixtures/synthetic_cybersecurity.parquet")
    report = corpus.profile(batch_size=2)
    assert report.row_count == 6 and not report.missing_required_columns
    texts = list(corpus.iter_text())
    tokenizer = BPETokenizer.train(texts, TokenizerConfig(vocab_size=320, min_frequency=1))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    assert tokenizer.decode(tokenizer.encode(texts[0])) == texts[0]

    blocks = list(chunk_token_stream(texts * 3, tokenizer, 16))
    loader = DataLoader(CausalTokenDataset(blocks[:4]), batch_size=2)
    batches = list(loader)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
    )
    model = CyberSLM(config)
    trainer = Trainer(
        model,
        TrainConfig(
            seed=4,
            device="cpu",
            batch_size=2,
            sequence_length=16,
            learning_rate=0.02,
            weight_decay=0.0,
            max_steps=30,
            warmup_steps=0,
            gradient_accumulation_steps=1,
            mixed_precision=False,
            log_every=50,
        ),
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=tmp_path / "checkpoints",
        log_path=tmp_path / "metrics.jsonl",
    )
    initial = trainer.evaluate([batches[0]])["validation_loss"]
    parameter_before = next(model.parameters()).detach().clone()
    losses = []
    for _ in range(30):
        losses.append(trainer.train_step([batches[0]]))
    assert not torch.equal(parameter_before, next(model.parameters()))
    assert losses[-1] < initial * 0.35
    assert all(torch.all(torch.isfinite(p.grad)) for p in model.parameters() if p.grad is not None)

    checkpoint = trainer.save("smoke.pt")
    restored = CyberSLM(config)
    resumed = Trainer(
        restored,
        trainer.config,
        tokenizer_fingerprint=tokenizer.fingerprint,
        checkpoint_dir=tmp_path / "restored",
        log_path=tmp_path / "restored.jsonl",
    )
    resumed.resume(checkpoint)
    assert resumed.state.step == trainer.state.step
    generated = generate_text(restored, tokenizer, "Fictional lab", max_new_tokens=3, top_k=5)
    assert isinstance(generated, str) and generated
