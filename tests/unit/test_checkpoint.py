from pathlib import Path

import torch

from cyberslm.models.transformer import CyberSLM, ModelConfig
from cyberslm.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_all_state(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        ffn_hidden_dim=32,
    )
    model = CyberSLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state={"step": 3},
        model_config=config,
        tokenizer_fingerprint="fixture",
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_model_config=config,
        expected_tokenizer_fingerprint="fixture",
    )
    assert state["step"] == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
