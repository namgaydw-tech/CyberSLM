import torch

from cyberslm.models.transformer import CyberSLM, ModelConfig, count_parameters
from cyberslm.tokenization.byte import ByteTokenizer
from cyberslm.training.evidence import AnswerOnlyDataset, EvidenceExample, pad_answer_only_batch


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "Cyber evidence: Δ"
    encoded = tokenizer.encode(text, bos=True, eos=True)
    assert tokenizer.vocab_size == 259
    assert tokenizer.decode(encoded) == text


def test_answer_only_mask_and_padding() -> None:
    tokenizer = ByteTokenizer()
    examples = [EvidenceExample("Question: 1? Answer: ", "one"), EvidenceExample("Q: ", "two")]
    dataset = AnswerOnlyDataset(examples, tokenizer=tokenizer, max_length=64)
    first_inputs, first_labels = dataset[0]
    prompt_length = len(tokenizer.encode(examples[0].prompt))
    assert torch.all(first_labels[:prompt_length] == -100)
    assert torch.all(first_labels[prompt_length:] != -100)
    inputs, labels = pad_answer_only_batch([dataset[0], dataset[1]])
    assert inputs.shape == labels.shape
    assert torch.all(labels[1, dataset[1][1].numel() :] == -100)
    assert first_inputs.numel() == first_labels.numel()


def test_cyberslm_35m_parameter_count() -> None:
    config = ModelConfig(
        vocab_size=259,
        max_seq_len=16,
        d_model=512,
        n_layers=10,
        n_heads=8,
        n_kv_heads=8,
        ffn_hidden_dim=1600,
        gradient_checkpointing=True,
    )
    count = count_parameters(CyberSLM(config))
    assert count == 35_205_120
    assert abs(count - 35_000_000) / 35_000_000 < 0.01


def test_eager_attention_forward_backward() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        ffn_hidden_dim=32,
        attention_implementation="eager",
    )
    model = CyberSLM(config)
    ids = torch.randint(0, 32, (1, 8))
    output = model(ids, ids)
    assert output.loss is not None
    output.loss.backward()
