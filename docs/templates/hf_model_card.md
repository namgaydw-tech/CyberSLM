---
library_name: pytorch
pipeline_tag: text-generation
tags:
  - custom-code
  - cybersecurity
  - coursework
license: other
---

# {{MODEL_NAME}}

CyberSLM is a custom decoder-only PyTorch language model created for Namgay D Wangchuk's
Design Thinking coursework at JAIN (Deemed-to-be University), assigned by Dr. Sujeet S.
Jagtap.

## Artifact status

- Status: **{{ARTIFACT_STATUS}}**
- Declared training stage: **{{TRAINING_STAGE}}**
- Parameters: **{{PARAMETER_COUNT}}**
- Checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Source revision: `{{CODE_REVISION}}`

{{TRAINING_DESCRIPTION}}

### Measured run results

- Stage A: 1,000 optimizer steps, 2,048,000 token exposures, final held-out loss
  **1.205688** (perplexity **3.3391**) on the frozen NIST validation documents.
- Stage B: 100 optimizer steps and 21,968 supervised answer-token exposures; best
  teacher-forced validation loss **0.058546**.
- Free-generation evaluation: validation exact match **32.14%** (28 examples) and
  cross-scenario exact match **32.66%** (199 examples). Cross-scenario numeric correctness
  was **56.28%**. These results reveal substantial template repetition and are not evidence
  of broad cybersecurity reasoning.

## Architecture

The model uses 10 decoder layers, hidden width 512, 8 attention heads, RoPE, RMSNorm,
SwiGLU width 1,600, tied embeddings, a 512-byte-token context, and a custom 259-token
UTF-8 byte vocabulary.

Token IDs are PAD=0, BOS=1, EOS=2, and each UTF-8 byte value is shifted by 3.

## Loading

This repository does **not** claim compatibility with `AutoModelForCausalLM`. It packages
custom PyTorch code and an inference-only state dictionary. Download the repository, install
PyTorch, and run:

```powershell
python load_model.py --prompt "Recorded-event evidence: " --max-new-tokens 64
```

The loader verifies the model configuration and tokenizer fingerprint before inference.

## Intended use

Educational, reproducibility, and constrained cybersecurity evidence experiments. This model
is not a production intrusion-detection system and must not be used as an autonomous attack
verdict engine.

## Limitations

- Byte-level generation is computationally inefficient compared with subword tokenizers.
- Source annotations and anomaly scores are not independently confirmed attack labels.
- Small or bounded training runs do not establish broad cybersecurity reasoning.
- The custom architecture is not compatible with Transformers `AutoModelForCausalLM`.
- The included GGUF imported and generated with Ollama 0.40.16 on Windows/CUDA. This verifies
  runtime execution, not exact native-logit parity or answer correctness.

## Files

- `pytorch_model.pt`: inference-only model state and verified metadata.
- `config.json`: custom architecture configuration.
- `tokenizer_config.json`: byte-token mapping and fingerprint.
- `modeling_cyberslm.py`, `tokenization_cyberslm.py`: custom implementation.
- `load_model.py`: tested native PyTorch loading example.
- `cyberslm-35m-stage-b-f16.gguf`: experimental F16 GGUF export.
- `checksums.json`: SHA-256 receipts for repository artifacts.

See the CyberSLM GitHub repository for dataset construction, training, validation, and full
coursework documentation.
