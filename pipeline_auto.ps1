# ============================================================
# pipeline_auto.ps1 - auto-run remaining DeepEye-SQL steps
# Usage:
#   .\pipeline_auto.ps1 -MaxItems 10                 # verify on first 10 questions (~30 CNY)
#   .\pipeline_auto.ps1 -ItemIds "1309,228,51"       # verify on specific question ids
#   .\pipeline_auto.ps1                              # run ALL remaining steps (1..6)
#   .\pipeline_auto.ps1 -MaxStage 2                  # stop after schema_linking
# ============================================================
param(
    [ValidateRange(1,6)]
    [int]$MaxStage = 6,
    [int]$MaxItems = 0,
    [string]$ItemIds = ""
)

$ErrorActionPreference = "Stop"

$ROOT   = "E:\Code\DeepEye-SQL"
$PY     = "C:\Users\LDM\.conda\envs\deepeye-sql\python.exe"
$CONFIG = "config/local/Qwen3.6-27B/config-bird-dev.toml"

Set-Location $ROOT
$env:CONFIG_PATH = $CONFIG

$LogFile = Join-Path $ROOT ("auto_pipeline_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Invoke-Step {
    param([string]$Name, [string]$Script, [string[]]$ArgList = @())
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host ("[{0}] START {1}" -f $Name, (Get-Date -Format "HH:mm:ss"))
    Write-Host ("=" * 60)
    Add-Content -Path $LogFile -Value ("[{0}] start {1}" -f $Name, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

    & $PY $Script @ArgList

    if ($LASTEXITCODE -ne 0) {
        $msg = "[{0}] FAILED (exit {1}) at {2}, stopping." -f $Name, $LASTEXITCODE, (Get-Date -Format "HH:mm:ss")
        Write-Host $msg -ForegroundColor Red
        Add-Content -Path $LogFile -Value $msg
        exit 1
    }
    $done = "[{0}] DONE {1}" -f $Name, (Get-Date -Format "HH:mm:ss")
    Write-Host $done -ForegroundColor Green
    Add-Content -Path $LogFile -Value ("[{0}] done {1}" -f $Name, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

# ---- wait for build_few_shot_index to finish (it is running in another terminal) ----
$Manifest = Join-Path $ROOT "workspace\runs\_shared\bird\train_full_index\manifest.json"
$WaitMinutes = 120
$Deadline = (Get-Date).AddMinutes($WaitMinutes)
Write-Host ("Waiting for build_few_shot_index: {0}" -f $Manifest)
while (-not (Test-Path $Manifest)) {
    if ((Get-Date) -gt $Deadline) {
        Write-Host ("Timeout after {0} min. build_few_shot_index may have failed, check manually." -f $WaitMinutes) -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds 30
    Write-Host ("  waiting... {0}" -f (Get-Date -Format "HH:mm:ss"))
}
Write-Host "build_few_shot_index done, starting remaining steps" -ForegroundColor Green

# ---- reset intermediate snapshots before a fresh small-batch run ----
# (dataset.snapshot / value_retrieval.snapshot are the full 1534-item base data and are KEPT)
$RunDir = Join-Path $ROOT "workspace\runs\bird-dev"
if ($ItemIds -ne "" -or $MaxItems -gt 0) {
    Write-Host "Resetting intermediate snapshots (keeping full base data)..." -ForegroundColor Yellow
    foreach ($name in @("few_shot_preparation", "schema_linking", "sql_generation", "sql_revision", "sql_selection")) {
        $snap = Join-Path $RunDir ($name + ".snapshot")
        $data = $snap + ".data"
        if (Test-Path $snap) { Remove-Item $snap -Force }
        if (Test-Path $data) { Remove-Item $data -Recurse -Force }
    }
}

# ---- run in order ----
$prepArgs = @()
if ($ItemIds -ne "")      { $prepArgs = @("--item_ids", $ItemIds, "--force") }
elseif ($MaxItems -gt 0)  { $prepArgs = @("--max_items", "$MaxItems", "--force") }
if ($MaxStage -ge 1) { Invoke-Step "1_few_shot_preparation" "runner\run_few_shot_preparation.py" $prepArgs }
if ($MaxStage -ge 2) { Invoke-Step "2_schema_linking"       "runner\run_schema_linking.py" }
if ($MaxStage -ge 3) { Invoke-Step "3_sql_generation"       "runner\run_sql_generation.py" }
if ($MaxStage -ge 4) { Invoke-Step "4_sql_revision"         "runner\run_sql_revision.py" }
if ($MaxStage -ge 5) { Invoke-Step "5_sql_selection"        "runner\run_sql_selection.py" }
if ($MaxStage -ge 6) { Invoke-Step "6_evaluation"           "runner\evaluation.py" }

Write-Host ""
Write-Host "ALL DONE" -ForegroundColor Green
