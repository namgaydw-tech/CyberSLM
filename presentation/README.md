# CyberSLM presentation pack

Student: **Namgay D Wangchuk**
USN: **25BTRCL300**
Course: **Design Thinking**
Instructor: **Dr. Sujeet S. Jagtap**
Institution: **JAIN (Deemed-to-be University)**
Presentation dates: **19-20 November 2026**

This directory is the presentation control point for the CyberSLM coursework demonstration.
It deliberately separates verified deliverables from work that is still running or blocked.

## Verified implementation snapshot

- Original CyberSLM-4M: 4,058,496 parameters; historical 300-step scratch-training run.
- CyberSLM-35M architecture: 35,205,120 trainable parameters.
- CyberSLM-35M engineering checks: CUDA forward/backward, optimizer update, checkpoint reload,
  and generation passed.
- Phase 4F evidence QA: 388 train, 28 validation, and 199 cross-scenario examples.
- Phase 4F validation: 615 of 615 answers and loss masks recomputed; zero retained duplicate
  prompt-answer pairs and zero cross-split event UUID overlap.
- Phase 6: Isolation Forest, novelty-mode LOF, and directed-graph window features implemented.
- Phase 7: deterministic and majority-answer baselines implemented.
- Native CLI and Streamlit demonstrations exist, but currently default to deterministic answers.

These facts do not establish production security detection, verified attack classification, or
broad cybersecurity reasoning.

## Required remote deliverables

| Deliverable | Verified value |
| --- | --- |
| GitHub main branch | `PENDING - remote main publication not yet verified` |
| GitHub commit SHA | `PENDING` |
| Hugging Face model repository | `https://huggingface.co/namgaydw-ai/CyberSLM-35M` |
| Hugging Face dataset repository | `https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA` |
| Stage A checkpoint | step 1,000; SHA-256 `cea167c1bd0d3709d3e912d00e26c885ccaae10202bb5622a96b6601d9f526b6` |
| Stage B checkpoint | step 100; SHA-256 `a86577065a5ab95ac76d82ffbefc3b4a60769ec493b03e2e3d3b22954adb9d94` |
| Ollama model name | `cyberslm-35m:latest` (Ollama 0.40.16 runtime verified) |

## Files

- `PRESENTATION_CHECKLIST.md`: pass/pending/blocked gate for each assignment requirement.
- `PRESENTATION_RUNBOOK.md`: exact Windows PowerShell presentation sequence and native fallback.

GitHub fields remain pending until the remote `main` revision is independently verified.
