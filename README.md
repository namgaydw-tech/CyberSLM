# CyberSLM-35M

CyberSLM is a scratch-built, 35.2-million-parameter decoder-only language model for evidence-grounded cybersecurity coursework. It was designed and trained locally by **Namgay D Wangchuk (25BTRCL300)** for the Design Thinking course at JAIN (Deemed-to-be University), assigned by **Dr. Sujeet S. Jagtap**.

> Research prototype, not a production detector. Generated answers, anomaly scores, and source annotations are not proof of an attack and must be checked against source evidence.

## Release status — 18 November 2026

| Component | Verified result |
| --- | --- |
| Architecture | 35,205,120 trainable parameters; forward/backward and checkpoint reload pass |
| Phase 4F evidence QA | 615 examples; 615/615 independently recomputed; all 68,027 source UUIDs accounted |
| Stage A pretraining | 1,000 steps; 2,048,000 token exposures; validation loss 1.205688; perplexity 3.3391 |
| Stage B adaptation | 100 steps; 21,968 answer-token exposures; best teacher-forced validation loss 0.058546 |
| Generation evaluation | validation EM 32.14%; cross-scenario EM 32.66%, numeric correctness 56.28% |
| GGUF / Ollama | F16 export imported and generated with Ollama 0.40.16; SHA-256 `d580042c919069ccf03bbe4327f8a60085f814a6ee5d432bde455bbf7b5aa831` |

The Stage B evaluation exposes a serious limitation: the model emitted only 11 unique predictions on each evaluation split and often repeated dominant training templates. Low teacher-forced loss does not demonstrate reliable free-form reasoning.

## Architecture

| Item | Value |
| --- | --- |
| Type | Decoder-only causal Transformer, trained from random initialization |
| Layers / width / heads | 10 / 512 / 8 |
| Feed-forward | SwiGLU, width 1,600 |
| Position / normalization | RoPE / RMSNorm |
| Attention | PyTorch SDPA with causal masking |
| Embeddings | Tied input/output embeddings |
| Tokenizer | UTF-8 bytes; 259 IDs: PAD=0, BOS=1, EOS=2, bytes=3–258 |
| Context | 512 byte tokens |
| Training aids | Gradient accumulation, gradient checkpointing, mixed precision |

```mermaid
flowchart LR
  A[Licensed NIST documents] --> B[Document-group split]
  B --> C[Stage A causal pretraining]
  D[Private Phase 4F evidence QA] --> E[Answer-only masks]
  C --> F[Stage B evidence adaptation]
  E --> F
  F --> G[Native PyTorch checkpoint]
  F --> H[F16 GGUF export]
  F --> I[Held-out generation evaluation]
```

## Data and provenance

Stage A uses ten public NIST publications listed in [`configs/stage_a_sources.json`](configs/stage_a_sources.json). Eight documents are assigned to training and two to validation at `document_id` level. No validation document contributes training bytes. The packed corpus contains 5,358,977 train tokens and 179,218 validation tokens; its Parquet SHA-256 is `b9e6bd8b526a0b3c0360c325c241eee45401e4af2c7268f51bb6f2fadc848ac0`.

Stage B uses the private Phase 4F evidence-QA artifact: 388 train, 28 validation, and 199 cross-scenario examples. It is not published because source release rights were not established. The public Hugging Face dataset is instead a six-row, unmistakably fictional schema/demo fixture using `.invalid` domains. It must not be presented as the training corpus.

Raw reports, production Parquet files, secrets, checkpoints, and experiment binaries are excluded from Git. Research runs record corpus, tokenizer, configuration, seed, code revision, and artifact hashes.

## Training results

### Stage A — licensed cybersecurity language pretraining

- Checkpoint SHA-256: `cea167c1bd0d3709d3e912d00e26c885ccaae10202bb5622a96b6601d9f526b6`
- 1,000 optimizer steps; 2,048,000 token exposures; 0.3822 effective corpus passes
- Validation loss 1.205688; byte perplexity 3.3391
- NVIDIA GeForce RTX 4050 Laptop GPU; 751.64 seconds; 2,724.71 tokens/second
- Peak allocated CUDA memory: 723,383,808 bytes

### Stage B — limited-data evidence adaptation

- Checkpoint SHA-256: `a86577065a5ab95ac76d82ffbefc3b4a60769ec493b03e2e3d3b22954adb9d94`
- 100 optimizer steps; 21,968 answer-token exposures; 1.0309 effective example epochs
- Duration 159.69 seconds; peak allocated CUDA memory 1,147,436,544 bytes
- Validation: 28 examples, 32.14% exact match and numeric correctness
- Cross-scenario: 199 examples, 32.66% exact match and 56.28% numeric correctness
- Unsupported-claim rate: 0% on both evaluated splits

The cross-scenario set was used during development and is not a pristine final benchmark. These are task-template metrics, not attack-detection precision, recall, F1, or attribution.

## Reproduce the bounded code path

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"

python scripts\build_stage_a_corpus.py --sources configs\stage_a_sources.json --output-dir data\stage_a\cyberslm_stage_a_nist_v1
python scripts\train_stage_a.py --config configs\training\cyberslm_35m_stage_a.yaml
python scripts\train_stage_b.py --config configs\training\cyberslm_35m_stage_b.yaml
python scripts\evaluate_checkpoint_qa.py --help
```

Review the corpus manifest and source/license record before training. Do not commit generated data. Run the quality gates before contributing:

```powershell
python -m ruff check src scripts tests
python -m mypy
python -m pytest -q
```

## Native model loading

The Hugging Face package contains custom PyTorch code and an inference state dictionary; it does **not** claim `AutoModelForCausalLM` compatibility:

```powershell
python load_model.py --prompt "Cybersecurity risk management requires" --max-new-tokens 64
```

## GGUF and Ollama

The source repository includes the exporter and a Modelfile, but not the 70.7 MB GGUF binary. Download `cyberslm-35m-stage-b-f16.gguf` from the model release, place it beside `ollama/Modelfile`, then run:

```powershell
ollama --version
ollama create cyberslm-35m -f .\ollama\Modelfile
ollama list
ollama run cyberslm-35m
```

Ollama 0.40.16 imported the GGUF as `cyberslm-35m:latest` and generated text on the RTX 4050. The first prompt continued with `and 1 repeated edge occurrences.` This confirms runtime execution, not semantic correctness or exact native-logit parity.

## Repository map

- `src/cyberslm/` — model, tokenizer, data, training, demo, and GGUF export modules
- `scripts/` — corpus, validation, training, evaluation, export, and packaging entry points
- `configs/` — frozen architecture, sources, and bounded training configurations
- `tests/` — unit and integration tests
- `docs/` — architecture, validation, experiment, and limitation records
- `presentation/` — verified demonstration checklist and runbook
- `app/` — local Streamlit evidence demonstration

## Public releases

- GitHub source: <https://github.com/namgaydw-tech/CyberSLM>
- Hugging Face model: <https://huggingface.co/namgaydw-ai/CyberSLM-35M>
- Public synthetic dataset: <https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA>

Consult the remote revision history and presentation checklist before describing a remote artifact as verified.

## Academic and safety limitations

- Phase 4F and earlier THEIA-derived artifacts are not released without recorded authorization.
- Synthetic fixtures are fictional and are never reported as observations.
- `ATTACK_CANDIDATE` annotations and anomaly scores are review signals, not confirmed attacks.
- Low language-model loss can coexist with memorized answer templates.
- This model must not autonomously block systems, attribute actors, or make incident verdicts.
- A new investigation-grouped, untouched test set and multiple-seed evaluation are required before stronger generalization claims.

See [`docs/experimental_results.md`](docs/experimental_results.md), [`docs/dataset_validation.md`](docs/dataset_validation.md), and [`docs/remaining_limitations.md`](docs/remaining_limitations.md) for the audit trail.
