---
library_name: pytorch
pipeline_tag: text-generation
tags:
  - custom-code
  - cybersecurity
  - coursework
  - byte-tokenizer
license: other
---

# {{MODEL_NAME}}

CyberSLM is a custom decoder-only PyTorch language model created for Namgay D Wangchuk's
Design Thinking coursework at JAIN (Deemed-to-be University), assigned by Dr. Sujeet S.
Jagtap.

## Artifact status

- Status: **{{ARTIFACT_STATUS}}**
- Training stage: **{{TRAINING_STAGE}}**
- Parameters: **{{PARAMETER_COUNT}}**
- Source checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Source revision: `{{CODE_REVISION}}`
- Training description: {{TRAINING_DESCRIPTION}}

The packaged weights are the improved structured-QA checkpoint trained from the Stage A step
1,000 checkpoint for 800 additional answer-only optimizer steps.

## Measured results

The new scenario-isolated frozen holdout contains 1,620 questions across 180 fictional
structured-evidence scenarios, balanced at 180 questions per task.

| Mode | Exact match |
|---|---:|
| Previous neural checkpoint | 49/1,620 = 3.02% |
| Improved neural checkpoint | 557/1,620 = 34.38% |
| Deterministic hybrid | 1,620/1,620 = 100% |

Improved neural numerical correctness is 178/900 = 19.78%. The hybrid recomputes answers from
structured fields using verified deterministic calculations. Its 100% result is **not** neural
model accuracy, and an unmodified model generation command does not provide the hybrid engine.

## Architecture and loading

The model has 10 decoder layers, hidden width 512, 8 attention heads, RoPE, RMSNorm, SwiGLU
width 1,600, tied embeddings, a 512-byte-token context, and a custom 259-token UTF-8 byte
vocabulary. Token IDs are PAD=0, BOS=1, EOS=2, and byte value `b` is ID `b+3`.

This package does not claim Transformers `AutoModelForCausalLM` compatibility. Install PyTorch
and run:

```powershell
python load_model.py --prompt "Structured cybersecurity evidence: " --max-new-tokens 32
```

## GGUF and Ollama

`cyberslm-35m-structured-byte-f16.gguf` uses a merge-free GPT-2 byte vocabulary so llama.cpp
produces the same byte IDs as native CyberSLM. Its SHA-256 is
`60d8e46194ea05b9be52c7d06538113a29bb4e8bb72db63c404637e64ca4ea40`.

Verification on Ollama 0.40.16 established identical input IDs on the diagnostic prompt,
near-identical top-ten first-token log probabilities, and 180/180 matching native/Ollama greedy
strings on a deterministic task-balanced sample. Runtime parity does not imply answer
correctness; official neural exact match remains 34.38%.

## Dataset information

The public dataset at
<https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA> is a six-row fictional
schema/demonstration fixture. It is not the private Phase 4F dataset or the generated 4,050-row
structured training corpus and must not be used to claim model accuracy.

## Intended use and limitations

This is an educational research prototype for reproducibility and constrained structured-QA
experiments. It is not a production intrusion detector or an autonomous attack-verdict engine.

- Aggregate neural exact match is only 34.38%; arithmetic remains particularly weak.
- The model can select the wrong task, hallucinate, or repeat text.
- Selected presentation examples are not an unbiased evaluation.
- Byte-level generation is less efficient than subword generation.
- Synthetic structured fixtures are not real incident observations.
- Source annotations and anomaly scores are not independently confirmed attack labels.

See <https://github.com/namgaydw-tech/CyberSLM> for construction, training, evaluation,
parity receipts, tests, and the presentation runbook.

## Files

- `pytorch_model.pt`: improved custom PyTorch inference state.
- `config.json`, `tokenizer_config.json`: verified architecture and byte-tokenizer metadata.
- `modeling_cyberslm.py`, `tokenization_cyberslm.py`: custom implementation.
- `load_model.py`: native loading example.
- `cyberslm-35m-structured-byte-f16.gguf`: parity-tested F16 GGUF.
- `training_summary.json`, `checksums.json`: training and artifact receipts.
