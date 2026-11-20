[CmdletBinding()]
param(
    [switch]$SkipLmStudio,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or not on PATH."
}
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

$Required = @(
    "cyberslm-35m-structured-byte:latest",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "all-minilm:latest"
)
$Available = (ollama list | Out-String)
foreach ($Model in $Required) {
    if ($Available -notmatch [regex]::Escape($Model)) {
        throw "Required Ollama model is missing: $Model"
    }
}

if (-not $SkipLmStudio -and (Get-Command lms -ErrorAction SilentlyContinue)) {
    if ((lms ps | Out-String) -notmatch "cyberslm-practical") {
        lms load qwen2.5-1.5b-instruct --identifier cyberslm-practical --gpu max --context-length 8192
    }
    if ((lms server status | Out-String) -match "not running") {
        lms server start --port 1234
    }
}

$env:PYTHONPATH = "src"
python -m streamlit run app\streamlit_app.py --server.port $Port
