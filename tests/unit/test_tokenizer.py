from pathlib import Path

from cyberslm.data.loader import ParquetCorpus
from cyberslm.tokenization.bpe import BPETokenizer, TokenizerConfig


def test_tokenizer_train_save_reload_and_security_identifiers(tmp_path: Path) -> None:
    texts = list(ParquetCorpus("data/fixtures/synthetic_cybersecurity.parquet").iter_text())
    tokenizer = BPETokenizer.train(
        texts,
        TokenizerConfig(vocab_size=320, min_frequency=1),
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    reloaded = BPETokenizer.load(path)
    sample = "CVE-2099-0001 demo.example.invalid 192.0.2.10 TCP SHA256"
    ids = reloaded.encode(sample, add_special_tokens=True)
    assert ids[0] == reloaded.bos_id
    assert ids[-1] == reloaded.eos_id
    assert reloaded.decode(ids) == sample
    assert reloaded.fingerprint == tokenizer.fingerprint
    assert reloaded.statistics([sample]).unknown_token_count == 0
