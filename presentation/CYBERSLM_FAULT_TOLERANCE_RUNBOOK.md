# Fault-tolerant professor demonstration

## Start

```powershell
cd "D:\Design thinking Cybersecurity SLM"
.\.venv\Scripts\Activate.ps1
.\presentation\Start-CyberSLM-Fault-Tolerant.ps1
```

Open `http://localhost:8501`. The sidebar shows both Qwen runtimes. Use **Original
CyberSLM-35M evidence mode** for the scratch contribution, **Practical conversational assistant**
for Qwen, and the **CyberSLM → Qwen Bridge** page for the attributed handoff.

## Verify before presenting

```powershell
.\presentation\Test-CyberSLM-Failover.ps1
python -m pytest -q tests\test_runtime_manager.py tests\test_two_model_bridge.py tests\test_chat_router_recovery.py tests\test_assistant.py
```

Expected supported scratch answers for the fictional three-event fixture:

1. `count unique event types` → reference `2`.
2. `count EVENT_READ` → reference `2`.
3. `is a confirmed attack established YES or NO` → reference `NO` (not established by counts).

These selected prompts are a demonstration, not an unbiased neural evaluation. The preserved frozen
test result is 557/1,620 = 34.38% neural exact match; the deterministic engine was 1,620/1,620.

## Demonstrate failover without damaging model registrations

1. Run the diagnostic. It deliberately targets an unused local port for the first Qwen attempt and
   shows LM Studio succeeding second.
2. The same diagnostic deliberately targets an unused port for scratch Ollama and shows the native
   checkpoint succeeding on CPU.
3. Do not delete or recreate either Ollama model. Do not stop a shared service during the live talk
   unless all other applications using it are closed.

## Private API for a remote host

```powershell
$env:CYBERSLM_API_TOKEN = '<long-random-secret>'
python scripts\serve_remote_api.py --host 127.0.0.1 --port 8765
```

Expose this only through an authorized HTTPS reverse proxy/tunnel that preserves the Bearer header.
Set Vercel's server-only `CYBERSLM_BACKEND_URL` and `CYBERSLM_BACKEND_TOKEN` to the corresponding
HTTPS URL and token. Never expose ports 11434 or 1234 directly.

## Honest limitations

The scratch model is reliable only for its bounded structured task. Qwen is externally pretrained.
RAG excerpts and generated prose require human review. If both Qwen services are down, deterministic
evidence remains available but conversational explanations do not. The Vercel preview has no live
model inference until a secured remote backend is authorized.
