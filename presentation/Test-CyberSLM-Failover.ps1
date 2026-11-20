[CmdletBinding()]
param([ValidateSet("cpu", "cuda")][string]$Device = "cpu")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONPATH = "src"

python scripts\diagnose_chat_stack.py --provider lm_studio --model cyberslm-practical
python scripts\diagnose_runtime_failover.py `
    --device $Device `
    --output outputs\runtime_failover_latest.json
