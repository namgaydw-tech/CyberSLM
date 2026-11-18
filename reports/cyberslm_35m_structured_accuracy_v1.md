# CyberSLM-35M structured evidence-QA accuracy experiment

Date: 2026-11-18
Status: completed experiment; neural 99% target not achieved; deterministic hybrid target achieved

## Result summary

| Mode | Frozen-test exact match | 95% Wilson interval | Numerical correctness |
|---|---:|---:|---:|
| Previous Stage B neural checkpoint | 49/1,620 = 3.02% | 2.30%–3.98% | 0/900 = 0.00% |
| Improved Stage B neural checkpoint | 557/1,620 = 34.38% | 32.11%–36.73% | 178/900 = 19.78% |
| Deterministic structured evidence engine | 1,620/1,620 = 100.00% | 99.76%–100.00% | 900/900 = 100.00% |

The hybrid result is deterministic recomputation from supplied structured facts. It is not
CyberSLM neural generation and is not available from an unmodified `ollama run` command.

Historical Phase 4F cross-scenario exact match was 32.66% on the earlier, already-exposed
benchmark. It is retained for provenance but is not directly comparable with this new frozen
holdout. The old checkpoint scored 3.02% when evaluated on the new holdout.

## Frozen evaluation methodology

- Dataset seed: 2031.
- 720 unique fictional scenarios, with all nine questions for one scenario assigned to exactly
  one split by `case_id`.
- Train: 450 scenarios / 4,050 questions.
- Validation: 90 scenarios / 810 questions.
- Frozen cross-scenario test: 180 scenarios / 1,620 questions, exactly 180 per task.
- No single-event-type scenario; event counts are unique within a scenario, avoiding extrema
  ties and majority-answer shortcuts.
- All references are recomputed deterministically from `facts_json` before scoring.
- The training configuration was fixed before the previous-model baseline was measured. The
  frozen test was not used for prompt selection, checkpoint selection, generation settings, or
  hyperparameters. The improved checkpoint was selected using validation loss only.
- The permitted event source was used only to validate the nine-event vocabulary and record a
  provenance receipt. No source rows or identifiers were copied. The source SHA-256 is
  `f2df8bc106bb755c336ed25b7294e3c5a91e3efd8ce10f986a6ac93e862699bc` (68,027 rows).
- Frozen test Parquet SHA-256:
  `736f14b71f2daffa401c159ef93a88f31b9b4e00506002646749f3ad3404c880`.

The dataset is schema-grounded synthetic evidence, not an observational dataset. Therefore the
experiment demonstrates narrow template/task behavior and deterministic calculation, not broad
cybersecurity incident understanding.

## Exact task definitions and improved neural results

Each task has 180 frozen-test examples.

| Task | Canonical answer | Neural exact match | Hybrid exact match |
|---|---|---:|---:|
| Event count | integer count for a named event type | 2.22% | 100% |
| Most frequent event type | event type with maximum count | 28.89% | 100% |
| Least frequent event type | event type with minimum count | 30.00% | 100% |
| Event percentage | named count / total, one decimal plus `%` | 0.00% | 100% |
| Distinct event types | number of event-type keys | 94.44% | 100% |
| Source and destination counts | `sources,destinations` | 1.11% | 100% |
| Directed graph edge count | supplied distinct directed-edge count | 1.11% | 100% |
| Temporal comparison | `first_type,last_type` | 51.67% | 100% |
| Unsupported attack verdict | `NO` when aggregates cannot prove an attack | 100.00% | 100% |

Improved neural error classes: 648 arithmetic errors, 254 wrong-task selections, 87
hallucinations, 74 repetitions, and 557 correct outputs. These counts cover all 1,620 rows.

## Training receipt

- Base: Stage A step 1,000,
  SHA-256 `cea167c1bd0d3709d3e912d00e26c885ccaae10202bb5622a96b6601d9f526b6`.
- Improved adaptation: 800 optimizer steps, batch size 2, gradient accumulation 4,
  learning rate `1e-4`, seed 2031.
- Answer-token exposures: 51,871; effective example epochs: 1.58025.
- Duration: 662.80 seconds; peak allocated CUDA memory: 1,148,613,632 bytes.
- First observed training loss: 7.02380; final training-window loss: 0.55004.
- Best validation loss: 0.56856 at step 800.
- Checkpoint:
  `outputs/cyberslm_35m_stage_b_structured_v1/checkpoints/final_step_000800.pt`.
- Checkpoint SHA-256:
  `72185b281de6baaf80003d8bdf5cbd4e2af9602f596a01b4308ef978f314f675`.
- Parameter count: 35,205,120.

## Prediction receipts

The previous and improved prediction Parquets include `evidence`, `question`, reference,
actual output, exact match, numerical correctness, error class, checkpoint identity, and
inference mode for every row:

- `outputs/cyberslm_35m_structured_baseline/evaluation/predictions.parquet`
- `outputs/cyberslm_35m_stage_b_structured_v1/evaluation/cross_scenario_test/predictions.parquet`

## GGUF and Ollama status

- The original F16 GGUF remains preserved as `ollama/cyberslm-35m-structured-f16.gguf`, SHA-256
  `103a2047a9032c67d4fdc479022ddb3b60d2db19f13c27a30695366b15cad744`.
- Root cause: its GGUF declared a SentencePiece `llama` tokenizer. llama.cpp encoded every ASCII
  space as the three UTF-8 bytes of `▁`, so an example native prompt had 278 tokens while the
  runtime received 308 different tokens. The weights were not the first divergence.
- The fixed exporter emits a merge-free GPT-2 byte vocabulary in the same ID order as
  CyberSLM: PAD=0, BOS=1, EOS=2, and byte `b` at `b+3`.
- Replacement F16 GGUF: `ollama/cyberslm-35m-structured-byte-f16.gguf`, 70,707,200 bytes.
- Replacement GGUF SHA-256:
  `60d8e46194ea05b9be52c7d06538113a29bb4e8bb72db63c404637e64ca4ea40`.
- Ollama 0.40.16 registered it as `cyberslm-35m-structured-byte:latest` while retaining
  `cyberslm-35m-structured:latest` and `cyberslm-35m:latest` as fallbacks.
- A diagnostic prompt produced 278 identical native/llama.cpp token IDs. Both runtimes selected
  first token ID 72 (`E`); their top-ten log probabilities differed by approximately `1e-3` or
  less.
- Full greedy strings matched on 180/180 deterministic task-balanced frozen-test prompts (20 per
  task). Native and Ollama were correct on the same 62/180 examples. This is a parity sample, not
  a replacement accuracy evaluation.

## Ollama demonstration (presentation-only)

The presentation uses three short examples selected from neural-correct frozen-test predictions.
They are not an unbiased sample and do not replace the 34.38% aggregate result.

1. Counts `EVENT_OPEN=22; EVENT_READ=8`. Question: count unique event types. Reference, native,
   and Ollama output: `2`.
2. Counts `EVENT_READ=2; EVENT_SENDTO=17; EVENT_WRITE=28`. Question: count `EVENT_READ`.
   Reference, native, and Ollama output: `2`.
3. Counts `EVENT_READ=15; EVENT_WRITE=25`. Question: is a confirmed attack established, yes or
   no? Reference, native, and Ollama output: `NO`.

Run all health, hash, registration, and inference checks with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\presentation\Run-CyberSLM-Demo.ps1
```

## Conclusion

The improved pure neural model did not achieve or approach 99% aggregate exact match. It
improved from 3.02% to 34.38% on the new holdout, but arithmetic and task routing remain major
limitations. The deterministic hybrid achieved 100% on the precisely defined structured tasks,
with a 99.76% lower Wilson bound, and is the only mode that clears the requested 99% point
target. The fixed byte-tokenizer GGUF passed the documented 180-prompt parity sample, but that
deployment repair does not improve or alter the measured neural accuracy.
