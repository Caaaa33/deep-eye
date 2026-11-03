param(
    [string]$Config = "config/local/Qwen3.6-27B/config-bird-dev.toml",
    [string]$Python = "python",
    [ValidateRange(1, 8)][int]$FromStep = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
$env:CONFIG_PATH = (Resolve-Path $Config).Path

$steps = @(
    @("dataset", "runner/preprocess_dataset.py"),
    @("vector_index", "runner/create_vector_db_parallel.py"),
    @("value_retrieval", "runner/run_value_retrieval.py"),
    @("few_shot_index", "runner/build_few_shot_index.py"),
    @("few_shot_preparation", "runner/run_few_shot_preparation.py"),
    @("schema_linking", "runner/run_schema_linking.py"),
    @("sql_generation", "runner/run_sql_generation.py"),
    @("sql_revision", "runner/run_sql_revision.py")
)

for ($i = $FromStep - 1; $i -lt $steps.Count; $i++) {
    $name, $entry = $steps[$i]
    Write-Host "[$($i + 1)/8] $name"
    & $Python $entry
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

$snapshot = & $Python -c "from app.config import get_config; print(get_config().sql_revision_config.save_path)"
if ($LASTEXITCODE -ne 0) { throw "Cannot resolve L3 snapshot path" }
if (-not (Test-Path -LiteralPath $snapshot -PathType Leaf)) { throw "L3 snapshot missing: $snapshot" }
Write-Host "L3 ready: $snapshot"
Write-Host "Freeze this snapshot and its .data directory before running the five L4 arms."
