# Runs the model-backed tracks of the full configuration one after another, then builds the
# ledger, the figures and the tables. Every stage is resumable: a stage that stopped part way
# skips the work units its run directory already holds when the script is started again.
# Run from the code directory in Windows PowerShell 5.1:
#     powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1
# Stages (log files under ..\logs, a logs directory beside the code directory):
#     capture, qwen2.5-1.5b   full_capture_qwen2.5-1.5b.log
#     live,    qwen2.5-1.5b   full_live_qwen2.5-1.5b.log
#     capture, qwen3-4b       full_capture_qwen3-4b.log
#     live,    qwen3-4b       full_live_qwen3-4b.log
#     ledger, figures, tables full_ledger.log, full_figures.log, full_tables.log

$ErrorActionPreference = "Stop"
$codeRoot = Split-Path -Parent $PSScriptRoot
Set-Location $codeRoot
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_XET = "1"
$config = "configs/full.yaml"
$logDir = Join-Path $codeRoot "..\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-FreeRamGb {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1MB, 2)
}

function Get-PlannedWorkers {
    $plan = python -c "import sys; sys.path.insert(0, '.'); from tasdiq.runners.config import load_config; from tasdiq.runners.workers import plan_workers; print(plan_workers(load_config('$config'), log=None).workers)"
    return [int]$plan
}

function Write-StageHeader {
    param([string]$Stage, [string]$LogPath)
    $line = "{0}  stage {1}  free RAM {2} GB  planned workers {3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Stage, (Get-FreeRamGb), (Get-PlannedWorkers)
    Write-Output $line
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

function Invoke-Stage {
    param([string]$Stage, [string]$LogPath, [string[]]$Arguments)
    Write-StageHeader -Stage $Stage -LogPath $LogPath
    $command = "python " + ($Arguments -join " ")
    Add-Content -Path $LogPath -Value $command -Encoding utf8
    # The interpreter runs under cmd.exe so that its standard output and error streams are
    # appended to the log as plain text; Windows PowerShell 5.1 would otherwise wrap every
    # line the model runtime prints on its error stream in an error record.
    & cmd.exe /c "$command >> `"$LogPath`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        $failure = "{0}  stage {1} stopped with exit code {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Stage, $LASTEXITCODE
        Add-Content -Path $LogPath -Value $failure -Encoding utf8
        throw $failure
    }
    $done = "{0}  stage {1} complete" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Stage
    Add-Content -Path $LogPath -Value $done -Encoding utf8
}

$stages = @(
    @{ Stage = "capture qwen2.5-1.5b"; Log = "full_capture_qwen2.5-1.5b.log"; Args = @("scripts/run_capture.py", "--config", $config, "--run-id", "full_capture", "--model", "qwen2.5-1.5b") },
    @{ Stage = "live qwen2.5-1.5b";    Log = "full_live_qwen2.5-1.5b.log";    Args = @("scripts/run_live.py", "--config", $config, "--run-id", "full_live", "--model", "qwen2.5-1.5b") },
    @{ Stage = "capture qwen3-4b";     Log = "full_capture_qwen3-4b.log";     Args = @("scripts/run_capture.py", "--config", $config, "--run-id", "full_capture", "--model", "qwen3-4b") },
    @{ Stage = "live qwen3-4b";        Log = "full_live_qwen3-4b.log";        Args = @("scripts/run_live.py", "--config", $config, "--run-id", "full_live", "--model", "qwen3-4b") },
    @{ Stage = "ledger";               Log = "full_ledger.log";               Args = @("scripts/build_ledger.py", "--config", $config) },
    @{ Stage = "figures";              Log = "full_figures.log";              Args = @("scripts/make_figures.py") },
    @{ Stage = "tables";               Log = "full_tables.log";               Args = @("scripts/fill_tables.py", "--refresh-templates") }
)

foreach ($entry in $stages) {
    $logPath = Join-Path $logDir $entry.Log
    Invoke-Stage -Stage $entry.Stage -LogPath $logPath -Arguments $entry.Args
}

Write-Output ("{0}  all stages complete" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
