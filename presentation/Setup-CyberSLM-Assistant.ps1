$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Install Ollama first: https://ollama.com/download/windows"
}
if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    py -3.13 -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install -e ".[assistant]"
& ".\.venv\Scripts\python.exe" -c "import pandas, pyarrow; print('Index serializers:', pandas.__version__, pyarrow.__version__)"
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull all-minilm
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" scripts\build_stage_a_corpus.py --sources configs\stage_a_sources.json --output-dir data\stage_a\cyberslm_stage_a_nist_v1
& ".\.venv\Scripts\python.exe" scripts\build_knowledge_index.py --config configs\rag_sources.json --output-dir data\rag\cyberslm_trusted_knowledge_v1
Write-Host "Setup complete. Run .\presentation\Start-CyberSLM-Assistant.ps1"
