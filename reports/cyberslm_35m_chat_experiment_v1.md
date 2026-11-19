# CyberSLM-35M-Chat experiment — honest release gate

Date: 19 November 2026
Branch: `codex/cyberslm-chat`

## Outcome

The work produced a genuinely trained 35,205,120-parameter conversational checkpoint, a separate
GGUF, exact Ollama/native runtime parity, and a tested Streamlit chat path. The checkpoint did
**not** meet the quality or retention gates and is therefore experimental, not a presentation or
Hugging Face release candidate. The verified structured model remains the presentation fallback.

| Gate | Result | Evidence |
| --- | --- | --- |
| Structured model preserved | PASS | No structured checkpoint/GGUF was replaced; `cyberslm-35m-structured-byte:latest` remains registered |
| Licensed/release-authorized corpus | PASS | Project-authored fictional conversations only; no private records or third-party corpus |
| Grouped train/validation/test split | PASS | 504/108/108 examples; 126/27/27 scenario groups; zero group overlap and zero duplicate prompts |
| Token/context validation | PASS | Existing byte tokenizer; maximum prompt-plus-answer length 447; 768-token memory probe passed but 512 retained |
| Controlled initialization | PASS | Stage A validation loss 2.1381 vs structured Stage B 2.8012; Stage A selected |
| Assistant-only SFT | PASS | Prompt labels masked; mixed precision, AdamW, cosine schedule, accumulation, clipping, validation, and checkpoints |
| At least 100 unseen conversations | PASS (execution) | 108 frozen test examples across nine categories and 27 unseen scenario groups |
| Neural chat quality | **FAIL** | 1/108 exact match; 31.48% relevance proxy; 31.48% factuality proxy; 43.52% termination |
| Independent human review | **BLOCKED** | Review packet created locally; no human ratings were invented |
| Ten-prompt acceptance dialogue | **FAIL** | Most responses were irrelevant project/checklist continuations; only one turn was plausibly on-scope |
| Structured neural retention | **FAIL** | 0/1,620 exact match and 0/900 numerical correctness on the frozen structured holdout |
| Deterministic hybrid retention | PASS | 1,620/1,620; this is verified calculation-engine accuracy, not neural generation |
| F16 GGUF export | PASS | SHA-256 `ff3f89fe05a21b75cc48754c7dc5f88c588b22f21cc551a4445270dc8f344ed9` |
| Native/Ollama parity | PASS | 10/10 exact greedy strings and token sequences at 64 generated tokens |
| Ollama registration | PASS | `cyberslm-35m-chat:latest`, Ollama 0.40.16 |
| Streamlit runtime | PASS | Headless startup succeeded and `/_stcore/health` returned `ok` |
| Public model/dataset update | **BLOCKED** | Failed quality and catastrophic-forgetting gates; no weights or dataset were uploaded |

## Data receipt

- Dataset ID: `cyberslm_chat_synthetic_v2`.
- License/provenance: project-authored synthetic fixture, release authorized by the project owner;
  no third-party conversation corpus and no private event records.
- Seed: 2041.
- Train/validation/test Parquet SHA-256:
  - train: `f74569b1f394ad449b0956dd1096fe6f365968490da204b7c18a2504e190e5db`
  - validation: `b2e6d0c3b2b737ea642f89d25f4f04f165e75762b6bb29f980fab084bcced467`
  - test: `bd504fd8cd8891e183fafbc7e958ea2e13fc17a3fd404f5afda8bdd920453b2c`
- Role format: `<|user|>\n`, `<|assistant|>\n`, and `\n<|end|>`.
- The corpus is generated locally and remains excluded from Git under the private-data policy.

## Training receipt

Stage A was selected rather than the structured Stage B checkpoint using assistant-only loss on
the held-out chat validation split. The initial bounded phase reached its best teacher-forced loss
of 0.59405 at step 125. A continuation reduced training loss to 0.03539 but worsened validation
loss to 0.75943 and exposed catastrophic forgetting.

- Evaluated continuation checkpoint: `step_000300.pt`.
- Checkpoint SHA-256: `9f5d12d5d96f926524dcebf1deaa3bae33a78d093fecdbd51a89f87518c4411a`.
- Initial Stage A SHA-256: `cea167c1bd0d3709d3e912d00e26c885ccaae10202bb5622a96b6601d9f526b6`.
- Optimizer: AdamW; learning rate 0.00003; weight decay 0.01.
- Batch size 2; four gradient-accumulation batches; mixed precision; gradient norm 1.0.
- Context remains 512 byte tokens. The 768-token forward/backward probe used only 371,195,904
  reserved CUDA bytes, but longer positions were not quality validated.

## Frozen evaluation and acceptance

The automated relevance/factuality values are keyword and number-consistency proxies, not human
judgments. The independent-human-review field remains pending. The fixed ten-turn dialogue showed
history-induced mode collapse: after a prior turn, the model repeatedly emitted patch-tracking or
password-hygiene project instructions for unrelated questions. The actual responses are preserved
in the local acceptance receipt and are intentionally not relabeled as successful examples.

The chat checkpoint also scored 0/1,620 on the original structured neural test, compared with the
preserved structured checkpoint's 557/1,620 (34.38%). The deterministic hybrid continues to score
1,620/1,620 because it performs verified calculations; it is separately labeled in the UI.

## Runtime and presentation decision

The experimental export is faithful to its PyTorch source: all 10 parity prompts matched exactly.
Parity does not imply correct answers. `ollama/Modelfile.chat` uses the exact application-supplied
role delimiters, a 512-token context, greedy decoding, and `<|end|>` stopping.

For the 19 November 2026 presentation, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\presentation\Run-CyberSLM-Demo.ps1
```

That command demonstrates the preserved `cyberslm-35m-structured-byte:latest` release. The chat
UI may be shown only as a failed experimental iteration with its warning visible; it must not be
described as a reliable conversational model.

## Required next research step

A credible next attempt needs a substantially broader, independently reviewed multi-turn corpus,
especially non-project dialogue histories, plus a mixed structured/chat replay objective and a
generation-based validation gate. More steps on the current corpus are not justified by these
results.
