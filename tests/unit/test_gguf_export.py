from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

gguf = pytest.importorskip("gguf")

from cyberslm.export.gguf import (  # noqa: E402
    byte_token_gguf_vocab,
    expected_tensor_mapping,
    export_checkpoint_to_gguf,
    validate_checkpoint_for_llama_gguf,
)
from cyberslm.models.transformer import CyberSLM, ModelConfig  # noqa: E402


def test_byte_token_vocab_preserves_ids() -> None:
    tokens, scores, token_types = byte_token_gguf_vocab()
    assert len(tokens) == len(scores) == len(token_types) == 259
    assert tokens[:3] == ["<pad>", "<s>", "</s>"]
    assert tokens[3] == "Ā"
    assert tokens[3 + ord(" ")] == "Ġ"
    assert tokens[3 + ord("A")] == "A"
    assert tokens[258] == "ÿ"
    assert token_types[:3] == [int(gguf.TokenType.CONTROL)] * 3
    assert set(token_types[3:]) == {int(gguf.TokenType.NORMAL)}


def test_tensor_mapping_and_checkpoint_shapes() -> None:
    config = ModelConfig(
        vocab_size=259,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        ffn_hidden_dim=64,
    )
    model = CyberSLM(config)
    payload = {"model_config": config.__dict__, "model": model.state_dict()}
    report = validate_checkpoint_for_llama_gguf(payload)
    mapping = expected_tensor_mapping(config.n_layers)
    assert report["tensor_count"] == 21
    assert set(mapping) == set(model.state_dict())
    assert mapping["layers.0.attention.q_proj.weight"] == "blk.0.attn_q.weight"
    assert mapping["layers.1.feed_forward.down.weight"] == "blk.1.ffn_down.weight"
    assert model.state_dict()["layers.0.attention.q_proj.weight"].shape == (32, 32)
    assert torch.equal(model.token_embedding.weight, model.lm_head.weight)


def test_tiny_gguf_round_trip_preserves_tensor_values(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=259,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        ffn_hidden_dim=64,
    )
    model = CyberSLM(config)
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.gguf"
    torch.save({"model_config": asdict(config), "model": model.state_dict()}, checkpoint)
    report = export_checkpoint_to_gguf(checkpoint, output, output_dtype="f32")
    reader = gguf.GGUFReader(output)
    assert report["status"] == "FORMAT_EXPORTED_RUNTIME_PARITY_NOT_YET_VERIFIED"
    assert len(reader.tensors) == len(expected_tensor_mapping(1))
    assert reader.get_field("tokenizer.ggml.model").contents() == "gpt2"
    tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    assert len(tokens) == 259 and tokens[3] == "Ā" and tokens[-1] == "ÿ"
    exported = reader.get_tensor(0).data
    torch.testing.assert_close(
        torch.from_numpy(exported.copy()), model.token_embedding.weight.detach()
    )
