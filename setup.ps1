[CmdletBinding()]
param(
    [switch]$NonInteractive,
    [ValidateSet('reswapper', 'seed-vc', 'rvc', 'deep-live-cam', 'w-okada')]
    [string[]]$Backend = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'This setup script targets Windows 11. See README for Linux development.' }
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable" }
}

Write-Host 'Unified Live — core setup (no AI models or drivers are downloaded)' -ForegroundColor Cyan
$windows = Get-CimInstance Win32_OperatingSystem
Write-Host "Windows: $($windows.Caption) / build $($windows.BuildNumber)"
$gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' })
if ($gpus.Count -eq 0) { Write-Warning 'No NVIDIA GPU detected. The mock cockpit still works on CPU.' }
foreach ($gpu in $gpus) { Write-Host "GPU: $($gpu.Name) / driver $($gpu.DriverVersion)" }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) {
    & py -3.11 -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.11 from python.org, then rerun setup.' }
    if (!(Test-Path '.venv\Scripts\python.exe')) { Invoke-Checked 'py' @('-3.11', '-m', 'venv', '.venv') }
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (!$python) { throw 'Python not found. Install 64-bit Python 3.11 with the py launcher.' }
    & python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The core setup currently requires Python 3.11.' }
    if (!(Test-Path '.venv\Scripts\python.exe')) { Invoke-Checked 'python' @('-m', 'venv', '.venv') }
}
$corePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Invoke-Checked $corePython @('-c', 'import struct, sys; sys.exit(0 if struct.calcsize("P") == 8 and sys.version_info[:2] == (3,11) else 1)')
Invoke-Checked $corePython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $corePython @('-m', 'pip', 'install', '-e', '.[dev]')

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    & ffmpeg -version | Select-Object -First 1
} else { Write-Warning 'FFmpeg not detected (not needed for mock live mode; some backends/training need it).' }
$obsCandidates = @(
    (Join-Path $env:ProgramFiles 'obs-studio\bin\64bit\obs64.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\obs-studio\bin\64bit\obs64.exe')
)
$obs = @($obsCandidates | Where-Object { Test-Path $_ })
if ($obs.Count) { Write-Host "OBS detected: $($obs[0])" }
else { Write-Warning 'OBS not found in standard locations. Install OBS separately for its virtual-camera driver.' }
Write-Host 'Checking playback devices for VB-CABLE / VoiceMeeter (names are hints; routing is verified when opened):'
Invoke-Checked $corePython @('-c', 'import sounddevice as sd; d=[(i,x["name"]) for i,x in enumerate(sd.query_devices()) if x["max_output_channels"] and any(s in x["name"].lower() for s in ("cable", "voicemeeter"))]; print(d if d else "No virtual cable playback endpoint detected")')

if (!$NonInteractive -and $Backend.Count -eq 0) {
    Write-Host 'Optional separate backend environments: reswapper, seed-vc, rvc, deep-live-cam, w-okada'
    Write-Host 'This only prepares empty venvs. Review docs/UPSTREAM_AUDIT.md before installing engines/models.'
    $selection = Read-Host 'Comma-separated backend names, or Enter to skip'
    if ($selection.Trim()) { $Backend = @($selection.Split(',') | ForEach-Object { $_.Trim() }) }
}
foreach ($engine in $Backend) {
    if ($engine -notin @('reswapper', 'seed-vc', 'rvc', 'deep-live-cam', 'w-okada')) {
        throw "Unknown backend: $engine"
    }
    $backendEnv = Join-Path $PSScriptRoot "runtime\$engine\venv"
    if (!(Test-Path (Join-Path $backendEnv 'Scripts\python.exe'))) {
        Invoke-Checked $corePython @('-m', 'venv', $backendEnv)
    }
    Write-Host "Prepared $backendEnv. Use the engine-specific Python version/dependencies in docs/BACKEND_INTEGRATION.md."
}
Write-Host 'Setup complete. Start the cockpit:' -ForegroundColor Green
Write-Host '.\.venv\Scripts\python -m unified_live --demo'
Write-Host 'For tests: .\.venv\Scripts\python -m pytest -q'
