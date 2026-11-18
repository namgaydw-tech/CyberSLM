from cyberslm.training.packed import PackedDocumentDataset


def test_packed_documents_have_shifted_fixed_blocks() -> None:
    dataset = PackedDocumentDataset(["fictional cybersecurity text " * 20], sequence_length=16)
    inputs, labels = dataset[0]
    assert inputs.shape == labels.shape == (16,)
    assert inputs[1:].tolist() == labels[:-1].tolist()
    assert dataset.unique_token_count > 16
