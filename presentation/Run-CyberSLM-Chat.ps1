$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

Write-Host "Checking verified structured fallback..."
ollama show cyberslm-35m-structured-byte | Out-Null

Write-Host "Checking experimental chat registration..."
$models = ollama list
if ($models -notmatch "cyberslm-35m-chat") {
    throw "cyberslm-35m-chat is not registered. Create it only after reviewing the failed quality report."
}

Write-Host "Starting CyberSLM UI at http://localhost:8501"
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
