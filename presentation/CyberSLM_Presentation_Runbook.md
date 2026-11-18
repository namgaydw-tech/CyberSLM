# CyberSLM presentation runbook

Presentation target: 19 November 2026
Prepared runtime: Windows PowerShell, Ollama 0.40.16

## One-command demonstration

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\presentation\Run-CyberSLM-Demo.ps1
```

The script performs bounded health checks, verifies checkpoint and GGUF SHA-256 values,
registers `cyberslm-35m-structured-byte` if needed, and runs the three prepared prompts. It exits
with an error if an artifact hash or expected answer differs.

## Presentation sequence

1. Open <https://github.com/namgaydw-tech/CyberSLM> and show the source, training configuration,
   tests, and reproducibility report.
2. Open <https://huggingface.co/namgaydw-ai/CyberSLM-35M> and show the improved custom PyTorch
   package and fixed byte-tokenizer GGUF.
3. Open <https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA> and state that it is a
   six-row fictional public demonstration fixture, not the private or generated training corpus.
4. Run the one-command demonstration above. Point out the checkpoint and GGUF hash PASS lines.
5. Show `ollama list`: the presentation model is
   `cyberslm-35m-structured-byte:latest`; the older registrations are fallbacks.
6. Read the three expected/actual result rows.

## Correct interpretation

- Previous neural checkpoint on the new frozen test: 49/1,620 = 3.02%.
- Improved neural checkpoint: 557/1,620 = 34.38%.
- Improved neural numerical correctness: 178/900 = 19.78%.
- Deterministic hybrid: 1,620/1,620 = 100%. This is calculation-engine accuracy, not neural
  generation accuracy.
- The three prepared prompts were selected from neural-correct frozen-test predictions for a
  stable live demonstration. They are not an unbiased evaluation sample.
- The fixed runtime achieved identical native/Ollama greedy strings on a deterministic,
  task-balanced 180-prompt parity sample.

## Fallback checks

```powershell
ollama --version
ollama list
ollama show cyberslm-35m-structured-byte --modelfile
Get-FileHash .\ollama\cyberslm-35m-structured-byte-f16.gguf -Algorithm SHA256
```

Expected GGUF SHA-256:
`60D8E46194EA05B9BE52C7D06538113A29BB4E8BB72DB63C404637E64CA4EA40`.

If the presentation model must be re-created:

```powershell
ollama create cyberslm-35m-structured-byte -f .\ollama\Modelfile.structured-byte
```
