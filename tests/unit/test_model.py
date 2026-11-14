import torch

from cyberslm.models.transformer import CyberSLM, ModelConfig, count_parameters


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
    )


def test_forward_shapes_loss_and_gradients() -> None:
    model = CyberSLM(tiny_config())
    inputs = torch.randint(0, 128, (2, 12))
    labels = torch.randint(0, 128, (2, 12))
    output = model(inputs, labels)
    assert output.logits.shape == (2, 12, 128)
    assert output.loss is not None
    output.loss.backward()
    assert count_parameters(model) > 0
    assert all(torch.all(torch.isfinite(p.grad)) for p in model.parameters() if p.grad is not None)


def test_causal_mask_blocks_future_information() -> None:
    torch.manual_seed(1)
    model = CyberSLM(tiny_config()).eval()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 99, 100]])
    with torch.no_grad():
        first = model(original).logits[:, :3]
        second = model(changed_future).logits[:, :3]
    torch.testing.assert_close(first, second)


def test_generation_runs() -> None:
    model = CyberSLM(tiny_config())
    output = model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, top_k=5)
    assert output.shape == (1, 5)
