$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or is not on PATH."
}
if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    throw "Missing .venv. Run presentation\Setup-CyberSLM-Assistant.ps1 first."
}
if (-not (Test-Path -LiteralPath ".\data\rag\cyberslm_trusted_knowledge_v1\manifest.json")) {
    throw "Missing knowledge index. Run presentation\Setup-CyberSLM-Assistant.ps1 first."
}

$RequiredModels = @("qwen2.5:1.5b-instruct-q4_K_M", "all-minilm:latest")
$Installed = ollama list
foreach ($Model in $RequiredModels) {
    if (-not ($Installed -match [regex]::Escape($Model))) {
        throw "Required Ollama model is not installed: $Model"
    }
}

$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
