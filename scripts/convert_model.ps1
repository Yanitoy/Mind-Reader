$ErrorActionPreference = "Stop"

param(
  [string]$InputModel = "expression_model.h5",
  [string]$OutputDir = "frontend/public/web_model"
)

if (-not (Test-Path $InputModel)) {
  throw "Input model not found: $InputModel"
}

if (-not (Get-Command tensorflowjs_converter -ErrorAction SilentlyContinue)) {
  throw "tensorflowjs_converter not found. Install with: pip install tensorflowjs"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

tensorflowjs_converter --input_format=keras $InputModel $OutputDir

Write-Host "Exported TF.js model to $OutputDir"
