param(
    [string]$Python = "py -3.11",
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment: $Venv"
    Invoke-Expression "$Python -m venv `"$Venv`""
}

$pythonExe = Join-Path $Venv "Scripts\python.exe"
Write-Host "Upgrading pip"
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing pitch_to_midi requirements"
& $pythonExe -m pip install -r requirements.txt

Write-Host "Done. Use:"
Write-Host "  .\$Venv\Scripts\python.exe .\sequence_pitch_pipeline.py --preview-only --write-previews 5"
Write-Host "  .\run_sequence_overnight.ps1"