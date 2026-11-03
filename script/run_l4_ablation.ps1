param(
    [string]$Config = "config/local/Qwen3.6-27B/config-bird-dev.toml",
    [string]$Python = "python",
    [string]$InputSnapshot = "",
    [string]$OutputRoot = "",
    [ValidateSet("A", "B", "C", "D", "E", "F")][string[]]$Arms = @("A", "B", "C", "D", "E", "F")
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
$env:CONFIG_PATH = (Resolve-Path $Config).Path

if (-not $InputSnapshot) {
    $InputSnapshot = & $Python -c "from app.config import get_config; print(get_config().sql_revision_config.save_path)"
    if ($LASTEXITCODE -ne 0) { throw "Cannot resolve L3 snapshot path" }
}
$InputSnapshot = (Resolve-Path -LiteralPath $InputSnapshot).Path
if (-not (Test-Path -LiteralPath ($InputSnapshot + ".data") -PathType Container)) {
    throw "L3 snapshot data directory missing: $($InputSnapshot).data"
}
if (-not $OutputRoot) { $OutputRoot = Join-Path (Split-Path $InputSnapshot -Parent) "l4_ablation" }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if ($OutputRoot -eq (Split-Path $InputSnapshot -Parent)) { throw "OutputRoot must differ from the L3 directory" }

if (@($Arms | Where-Object { $_ -in @('B', 'D') }).Count -gt 0) {
    $judgeReady = & $Python -c "from app.config import get_config; print('1' if get_config().sql_selection_config.judge_llm is not None else '0')"
    if ($LASTEXITCODE -ne 0 -or $judgeReady -ne '1') {
        throw "Arms B/D require judge_llm_profile or [sql_selection.judge_llm] in the config"
    }
}

$inputHash = (Get-FileHash -LiteralPath $InputSnapshot -Algorithm SHA256).Hash
$configHash = (Get-FileHash -LiteralPath $env:CONFIG_PATH -Algorithm SHA256).Hash
foreach ($arm in $Arms) {
    $armDir = Join-Path $OutputRoot "arm_$arm"
    $outputSnapshot = Join-Path $armDir "sql_selection.snapshot"
    if ([System.IO.Path]::GetFullPath($outputSnapshot) -eq $InputSnapshot) { throw "Output would overwrite L3" }
    $marker = Join-Path $armDir "ablation_run.json"
    $identity = [ordered]@{ arm = $arm; input_snapshot = $InputSnapshot; input_sha256 = $inputHash; config_sha256 = $configHash }
    if (Test-Path -LiteralPath $marker) {
        $previous = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
        if ($previous.arm -ne $arm -or $previous.input_snapshot -ne $InputSnapshot -or $previous.input_sha256 -ne $inputHash -or $previous.config_sha256 -ne $configHash) {
            throw "Existing arm $arm output belongs to a different input or config; choose a new OutputRoot"
        }
    } elseif (Test-Path -LiteralPath $outputSnapshot) {
        throw "Existing arm $arm snapshot has no ablation_run.json; choose a new OutputRoot"
    }
    New-Item -ItemType Directory -Path $armDir -Force | Out-Null
    $identity | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    Write-Host "Running arm $arm from $InputSnapshot -> $outputSnapshot"
    & $Python runner/run_sql_selection.py --config $env:CONFIG_PATH --arm $arm --input_snapshot $InputSnapshot --output_snapshot $outputSnapshot
    if ($LASTEXITCODE -ne 0) { throw "L4 arm $arm failed with exit code $LASTEXITCODE" }
    if ((Get-FileHash -LiteralPath $InputSnapshot -Algorithm SHA256).Hash -ne $inputHash) { throw "L3 manifest changed during arm $arm" }
    & $Python runner/evaluation.py --snapshot_path $outputSnapshot
    if ($LASTEXITCODE -ne 0) { throw "Evaluation for arm $arm failed with exit code $LASTEXITCODE" }
}
Write-Host "L4 ablation completed under $OutputRoot"
