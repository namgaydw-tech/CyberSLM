# CyberSLM presentation runbook

All commands below are Windows PowerShell commands. Start from a fresh terminal. Do not run an
Ollama or Hugging Face command with a placeholder path or URL.

## 1. Open the project and environment

```powershell
Set-Location -LiteralPath "D:\Design thinking Cybersecurity SLM"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
git branch --show-current
git status --short
```

Expected branch before publication: `codex/phase4f-cyberslm35m`.

## 2. Show the tested architecture

```powershell
python -c "from cyberslm.models.configuration import load_model_config; from cyberslm.models.transformer import CyberSLM, count_parameters; c=load_model_config('configs/model/cyberslm_35m.yaml'); print(c); print('parameters=', count_parameters(CyberSLM(c)))"
```

Verified parameter count: `35,205,120`.

## 3. Verify Phase 4F privately

The source path below is private and must not be pasted into public documentation or uploaded.

```powershell
$source = "CyberSLM_Phase4_Evidence\runs\phase4a_provenance\source_event_files\datasets_train_data\processed\theia_standardized.parquet"
$phase4f = "CyberSLM_Phase4_Evidence\runs\phase4f_evidence_qa_v2"
python scripts\validate_phase4f.py --source $source --dataset-dir $phase4f
```

Expected verified result from 18 Nov 2026: `status=PASS`, `qa_rows=615`, and
`answers_recomputed=615`.

## 4. Build the Stage A NIST corpus

This command downloads the NIST publications listed in `configs/stage_a_sources.json`. Verify the
individual-document markings and final manifest before training or publication.

```powershell
python scripts\build_stage_a_corpus.py `
  --sources configs\stage_a_sources.json `
  --output-dir data\stage_a\cyberslm_stage_a_nist_v1
```

Before proceeding, inspect:

```powershell
Get-Content data\stage_a\cyberslm_stage_a_nist_v1\manifest.json
```

```powershell
python scripts\train_stage_a.py --config configs\training\cyberslm_35m_stage_a.yaml
python scripts\train_stage_b.py --config configs\training\cyberslm_35m_stage_b.yaml
python scripts\evaluate_checkpoint_qa.py --help
```

Recorded Stage A result: 1,000 steps, 2,048,000 token exposures, validation loss 1.205688.
Recorded Stage B generation result: 32.14% validation exact match and 32.66%
cross-scenario exact match. The low free-generation score and repeated templates must be shown.

## 5. Show current native fallback

This command uses an unmistakably fictional fixture and deterministic calculations. It is not
CyberSLM-35M neural inference.

```powershell
python scripts\run_demo.py --question "Which event type is most frequent?"
```

## 6. Launch Streamlit

Install the optional analysis/demo dependencies if they are not already present:

```powershell
python -m pip install -r requirements-analysis.txt
$env:PYTHONPATH = "src"
streamlit run app\streamlit_app.py
```

Current status: the app displays deterministic evidence calculations. A trained-checkpoint mode
must not be claimed until it has been added and tested.

## 7. Show Hugging Face artifacts

```text
Model repository:   https://huggingface.co/namgaydw-ai/CyberSLM-35M
Dataset repository: https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA
```

Both repositories were downloaded into a fresh temporary directory on 18 November 2026. The
downloaded native model restored and generated text; the GGUF hash matched the release receipt.
The dataset manifest confirmed six fictional rows and `not_the_phase4f_dataset=true`.

## 8. Ollama compatibility and import

First inspect the installation:

```powershell
ollama --version
ollama list
```

The custom PyTorch `.pt` checkpoint is not itself an Ollama model. The F16 GGUF hash is
`d580042c919069ccf03bbe4327f8a60085f814a6ee5d432bde455bbf7b5aa831`.

```powershell
Test-Path -LiteralPath ".\ollama\cyberslm-35m-stage-b-f16.gguf"
Get-FileHash -Algorithm SHA256 ".\ollama\cyberslm-35m-stage-b-f16.gguf"
ollama create cyberslm-35m -f .\ollama\Modelfile
ollama list
ollama run cyberslm-35m
```

Do not create a `Modelfile` that points to a `.pt` file, renamed tensor dump, or unrelated model.
If conversion remains incompatible, state that Ollama is blocked and use the native fallback in
section 5.

## 9. Three-question demonstration

Use the prompts in `PRESENTATION_CHECKLIST.md`. For each one:

1. Show the structured evidence.
2. Show the deterministic expected calculation.
3. Select the exact checkpoint or Ollama model being evaluated.
4. Record the actual model output without editing it.
5. Explain numerical errors, repetition, unsupported claims, and limitations.

Recorded Ollama outputs are in `PRESENTATION_CHECKLIST.md`. They are incorrect and repetitive;
show them unedited as evidence of the model's limited generalization.

## 10. Show GitHub main

```text
GitHub main URL:  PENDING
Published commit: PENDING
```

Before presenting a commit as published, verify it from a fresh fetch:

```powershell
git fetch origin
git rev-parse origin/main
git log -1 --oneline origin/main
```

## 11. Closing statement

Use this wording unless later verified results justify an update:

> CyberSLM is a scratch-built coursework language-model and evidence-processing implementation.
> It is not a production detector. Its numerical checks, anomaly scores, and generated language
> must be validated against source evidence. Any incomplete hosting or Ollama item is identified
> explicitly rather than being simulated.
