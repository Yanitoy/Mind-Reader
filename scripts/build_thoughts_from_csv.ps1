param(
  [string]$InputPath = "backend/train.csv",
  [string]$OutputPath = "frontend/public/thoughts.json",
  [int]$MaxPerLabel = 60,
  [int]$MaxLength = 140,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$labelMap = @{
  "0" = "sad"
  "1" = "happy"
  "2" = "happy"
  "3" = "angry"
  "4" = "fear"
  "5" = "surprise"
  "6" = "neutral"
  "7" = "disgust"
}

$textMap = @{
  "sadness" = "sad"
  "sad" = "sad"
  "anger" = "angry"
  "angry" = "angry"
  "fear" = "fear"
  "joy" = "happy"
  "happy" = "happy"
  "love" = "happy"
  "surprise" = "surprise"
  "disgust" = "disgust"
  "neutral" = "neutral"
}

$emotions = @("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
$buckets = @{}
foreach ($emotion in $emotions) {
  $buckets[$emotion] = New-Object System.Collections.Generic.List[string]
}

if (-not (Test-Path $InputPath)) {
  throw "CSV not found: $InputPath"
}

$rows = Import-Csv -Path $InputPath
foreach ($row in $rows) {
  $text = $row.text
  if (-not $text) { $text = $row.dialogue }
  if (-not $text) { $text = $row.sentence }
  if (-not $text) { $text = $row.utterance }
  if (-not $text) { continue }
  $text = $text.Trim()
  if ($text.Length -gt $MaxLength) { continue }

  $rawLabel = $row.label
  if (-not $rawLabel) { $rawLabel = $row.emotion }
  if (-not $rawLabel) { continue }

  $emotion = $null
  $labelKey = $rawLabel.ToString().Trim()
  if ($labelMap.ContainsKey($labelKey)) {
    $emotion = $labelMap[$labelKey]
  } else {
    $normalized = $labelKey.ToLowerInvariant()
    if ($textMap.ContainsKey($normalized)) {
      $emotion = $textMap[$normalized]
    }
  }

  if ($emotion -and $buckets.ContainsKey($emotion)) {
    $buckets[$emotion].Add($text)
  }
}

$rand = New-Object System.Random $Seed
$output = @{}
foreach ($emotion in $emotions) {
  $unique = $buckets[$emotion] | Sort-Object -Unique
  $shuffled = $unique | Sort-Object { $rand.Next() }
  if ($shuffled.Count -gt $MaxPerLabel) {
    $shuffled = $shuffled[0..($MaxPerLabel - 1)]
  }
  $output[$emotion] = $shuffled
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$json = $output | ConvertTo-Json -Depth 3
Set-Content -Path $OutputPath -Value $json
Write-Host "Wrote $OutputPath"
