param(
    [switch]$SkipCreate
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$modelName = 'cyberslm-35m-structured-byte'
$expectedCheckpointHash = '72185B281DE6BAAF80003D8BDF5CBD4E2AF9602F596A01B4308EF978F314F675'
$expectedGgufHash = '60D8E46194EA05B9BE52C7D06538113A29BB4E8BB72DB63C404637E64CA4EA40'
$checkpoint = Join-Path $repoRoot 'outputs\cyberslm_35m_stage_b_structured_v1\checkpoints\final_step_000800.pt'
$gguf = Join-Path $repoRoot 'ollama\cyberslm-35m-structured-byte-f16.gguf'
$modelfile = Join-Path $repoRoot 'ollama\Modelfile.structured-byte'
$demoFile = Join-Path $PSScriptRoot 'cyberslm_ollama_demo.json'

$ollamaCommand = Get-Command ollama.exe -ErrorAction SilentlyContinue
if ($ollamaCommand) {
    $ollama = $ollamaCommand.Source
} else {
    $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
}
if (-not (Test-Path -LiteralPath $ollama)) {
    throw "Ollama executable not found at $ollama"
}

Write-Host 'CyberSLM presentation health check' -ForegroundColor Cyan
Write-Host "GitHub: https://github.com/namgaydw-tech/CyberSLM"
Write-Host "Hugging Face model: https://huggingface.co/namgaydw-ai/CyberSLM-35M"
Write-Host "Hugging Face dataset: https://huggingface.co/datasets/namgaydw-ai/CyberSLM-Evidence-QA"
& $ollama --version

try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null
} catch {
    Write-Host 'Starting Ollama service...'
    Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch {
            # Continue the bounded startup poll.
        }
    }
    if (-not $ready) {
        throw 'Ollama service did not become ready within 10 seconds.'
    }
}

foreach ($artifact in @(
    @{ Path = $checkpoint; Expected = $expectedCheckpointHash; Label = 'checkpoint' },
    @{ Path = $gguf; Expected = $expectedGgufHash; Label = 'GGUF' }
)) {
    if (-not (Test-Path -LiteralPath $artifact.Path)) {
        throw "Missing $($artifact.Label): $($artifact.Path)"
    }
    $actualHash = (Get-FileHash -LiteralPath $artifact.Path -Algorithm SHA256).Hash
    if ($actualHash -ne $artifact.Expected) {
        throw "$($artifact.Label) SHA-256 mismatch: $actualHash"
    }
    Write-Host "$($artifact.Label) SHA-256 PASS: $actualHash" -ForegroundColor Green
}

if (-not $SkipCreate) {
    $registered = & $ollama list | Select-String -SimpleMatch "$modelName`:latest"
    if (-not $registered) {
        & $ollama create $modelName -f $modelfile
        if ($LASTEXITCODE -ne 0) {
            throw 'Ollama model creation failed.'
        }
    }
}

$results = @()
foreach ($demo in (Get-Content -Raw -LiteralPath $demoFile | ConvertFrom-Json)) {
    $body = @{
        model = $modelName
        prompt = $demo.prompt
        stream = $false
        options = @{
            temperature = 0
            top_k = 1
            top_p = 1
            min_p = 0
            repeat_penalty = 1
            num_ctx = 512
            num_predict = 32
        }
    } | ConvertTo-Json -Depth 5
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' `
        -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 120
    $actual = $response.response.Trim()
    $passed = $actual -ceq $demo.reference_answer
    $results += [pscustomobject]@{
        Task = $demo.task
        Expected = $demo.reference_answer
        Actual = $actual
        Status = if ($passed) { 'PASS' } else { 'FAIL' }
    }
}

$results | Format-Table -AutoSize
if ($results.Status -contains 'FAIL') {
    throw 'One or more prepared demonstrations failed.'
}

Write-Host 'Demo PASS. Selected examples are presentation-only.' -ForegroundColor Green
Write-Host 'Frozen-test neural EM: 557/1,620 = 34.38%.'
Write-Host 'Deterministic hybrid EM: 1,620/1,620 = 100%; this is not neural accuracy.'
