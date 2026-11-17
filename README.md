# CyberSLM

Research-grade, dataset-ready infrastructure for training a compact cybersecurity-specialized language model from random weights. The repository currently provides verified foundations—not a trained model, a real cybersecurity corpus, or benchmark claims.

## Status

| Area | Status |
|---|---|
| Streaming Parquet inspection, schema validation, quality report | Implemented and tested |
| Document-grouped and chronological split assignment | Implemented and tested |
| Byte-level BPE training, serialization, statistics | Implemented and tested on fictional data |
| Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, causal attention) | Implemented and tested |
| AdamW training, scheduling, clipping, accumulation, AMP, JSONL metrics | Implemented and CPU-tested |
| Compatible checkpoint save/resume and text generation | Implemented and tested |
| Real dataset integration and large-scale pretraining | Pending authorized dataset and hardware |
| Cybersecurity benchmark results, SFT, retrieval | Planned; no claims are made |

No project license has been added because ownership and license choice have not been confirmed. Until the owner selects a license, reuse rights are not granted by this repository.

## Quick start

Supported Python: 3.10–3.13. CI uses Python 3.11 and CPU-only tests.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/create_synthetic_fixture.py
python -m pytest -m "not slow"
python -m pytest tests/integration/test_smoke_pipeline.py -m slow -v
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/create_synthetic_fixture.py
make check
```

The fixture is entirely fictional. Its reserved domains/IP ranges and future-dated identifiers are only for software tests.

## Validate delivered Parquet data

Keep private files under `data/raw/`; Git ignores them. First inspect without modifying the source:

```powershell
cyberslm-data data/raw/*.parquet --batch-size 4096
cyberslm-data data/raw/*.parquet --text-column text --document-id-column document_id --fail-on-quality-issues
```

If column names differ, map `--text-column` and `--document-id-column`. Before training, record source licenses, confirm document IDs are stable, create group-level splits, and train the tokenizer only on the resulting training split.

## Explicit training flow

Training never starts automatically. A real run requires reviewed split files and an explicitly trained tokenizer:

```powershell
cyberslm-tokenizer data/splits/train/*.parquet --output experiments/run-001/tokenizer.json --vocab-size 16000
cyberslm-train --data data/splits/train/*.parquet --tokenizer experiments/run-001/tokenizer.json --model-config configs/model/approx_30m.yaml --training-config configs/training/pretrain.yaml --output-dir experiments/run-001
cyberslm-generate --checkpoint experiments/run-001/checkpoints/final.pt --tokenizer experiments/run-001/tokenizer.json --prompt "Explain the evidence:"
```

The `approx_*` filenames are experiment budget targets only. `configured_parameter_count` calculates actual counts; the final architecture must be chosen from dataset token counts, compute, and ablation evidence.

| Candidate configuration | Calculated parameters |
|---|---:|
| `approx_30m.yaml` | 33,890,816 |
| `approx_50m.yaml` | 50,062,720 |
| `approx_100m.yaml` | 102,450,432 |

## Architecture and layout

```text
configs/                    model, training, tokenizer, and data settings
data/                       ignored raw/processed/splits plus fictional fixture
src/cyberslm/
  data/                     lazy Parquet loading, profiling, safe splits
  tokenization/             byte-level BPE and token statistics
  models/                   compact causal Transformer
  training/                 dataset packing, trainer, checkpoints, CLI
  evaluation/               LM metrics and benchmark contracts
  inference/                autoregressive generation and CLI
  retrieval/                attributable evidence interface (future)
tests/                      unit and end-to-end CPU smoke tests
docs/                       architecture, schema, methodology, cards
```

See [architecture](docs/architecture.md), [dataset schema](docs/data-schema.md), [research methodology](docs/research-methodology.md), and [security policy](docs/security-and-data-handling.md).

## Research context and boundaries

CyberPal 2.0 motivates expert-guided, evidence-grounded cybersecurity data and evaluation, but CyberSLM does not copy its architecture or claim to reproduce its results. The Awesome Annual Security Reports repository is a discovery index; each original publisher's license must be checked before content use or redistribution. No external dataset is downloaded by this project.

- Levi et al., “Toward Cybersecurity-Expert Small Language Models,” arXiv:2510.14113v2 (2026), CC BY 4.0.
- Wilson, “Awesome Annual Security Reports,” discovery index; repository MIT license does not automatically license linked reports.

## Reproducibility and limitations

Every run should retain configuration snapshots, code revision, dataset manifest/hash, split assignment, tokenizer fingerprint, seed, metric logs, and checkpoints. The current trainer restores model, optimizer, scheduler, and step/token state. Distributed training, sharded checkpoints, dataset versioning, SFT, production safety controls, and retrieval implementations are not yet present.

