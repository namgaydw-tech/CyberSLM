# CyberSLM Assistant presentation runbook

## Ollama: recommended presentation path

From the repository root:

```powershell
.\presentation\Start-CyberSLM-Assistant.ps1
```

Open the displayed local Streamlit URL. Confirm the sidebar says:

- capability: `Practical conversational assistant`;
- runtime: `Ollama`;
- model: `qwen2.5:1.5b-instruct-q4_K_M`;
- trusted knowledge retrieval: enabled.

Use these prompts in order:

1. `Hi CyberSLM.`
2. `Explain ransomware in detail.`
3. `How can machine learning help detect it?`
4. `Give me a Python cybersecurity project idea.`
5. `Explain how to implement that project.`
6. `Summarize our discussion.`

The answers are generated at runtime, so wording varies. Show latency and retrieved source links.
Do not describe the selected examples as an unbiased evaluation.

## Original scratch-built model

Switch capability to `Original CyberSLM-35M evidence mode`. Explain that this is the genuine
scratch-built 35.2M Transformer and that the verified deterministic calculation is displayed
separately from its neural output. Keep `cyberslm-35m-structured-byte:latest` registered.

## LM Studio alternative

First-time model download (LM Studio CLI 1.3.3 or later):

```powershell
lms get "Qwen/Qwen2.5-1.5B-Instruct-GGUF@Q4_K_M" --gguf -y
lms load qwen2.5-1.5b-instruct --identifier cyberslm-practical --gpu max --context-length 8192
lms server start --port 1234
Invoke-RestMethod http://127.0.0.1:1234/v1/models
```

In Streamlit, select `LM Studio`. Retrieval and policy routing still occur in the CyberSLM Python
application; LM Studio performs answer generation through its OpenAI-compatible local API.

## Health checks

```powershell
ollama --version
ollama list
ollama ps
lms server status
lms ps
Get-FileHash "$env:USERPROFILE\.lmstudio\models\Qwen\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" -Algorithm SHA256
```

Expected practical GGUF SHA-256:
`183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4`.

## Interpretation to state aloud

The practical assistant is Qwen2.5-1.5B-Instruct plus CyberSLM retrieval, routing, history, and UI.
It is not the scratch-built CyberSLM-35M. The scratch model remains the original research
contribution. Retrieved sources support investigation but do not make every generated sentence
correct, and a security event never proves an attack without corroborating evidence.
