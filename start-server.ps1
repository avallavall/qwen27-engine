# Start the q27 engine server: Qwen3.8-27B on the two cards, OpenAI-compatible API (like llama-server).
# Usage:  powershell -File start-server.ps1 [-Port 8081] [-Bind 127.0.0.1] [-Ctx 0] [-Effort medium] [-Kv q8_0]
#         -Bind 0.0.0.0 makes it reachable over the LAN / VPN.
#         The API key comes from -ApiKey or the environment variable Q27_API_KEY. Without one the server is open.
# Stop it with Ctrl+C in this window.
param(
  [int]$Port = 8081,
  [string]$Bind = "127.0.0.1",
  [int]$Ctx = 0,                 # 0 = the largest context that fits (vision reserve included)
  [string]$ApiKey = $env:Q27_API_KEY,
  [string]$Effort = "medium",    # reasoning effort default: xhigh, medium or low
  [string]$Kv = "q8_0",          # KV cache type: q8_0 (default) or f16
  [string]$Model = "",           # the GGUF (see Find-ModelFile below)
  [string]$Mmproj = "",          # image encoder of the same model; "none" = text only
  [switch]$NoVision,
  [string]$LogDir = ""           # optional: write every request and result as JSON files here
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# Model files: -Model / -Mmproj, else $env:Q27_MODEL / $env:Q27_MMPROJ, else models\ in this repo, else a
# qwen38_27\models\Qwen3.8-27B folder next to this repo.
function Find-ModelFile([string]$given, [string]$envName, [string]$file) {
  if ($given) { return $given }
  $e = [Environment]::GetEnvironmentVariable($envName)
  if ($e) { return $e }
  foreach ($d in @((Join-Path $PSScriptRoot "models"), (Join-Path $PSScriptRoot "..\qwen38_27\models\Qwen3.8-27B"))) {
    $p = Join-Path $d $file
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }
  return (Join-Path (Join-Path $PSScriptRoot "models") $file)
}
$Model = Find-ModelFile $Model "Q27_MODEL" "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
if ($NoVision -or $Mmproj -eq "none") { $Mmproj = "" } else { $Mmproj = Find-ModelFile $Mmproj "Q27_MMPROJ" "mmproj-Qwen3.8-27B-BF16.gguf" }
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"   # same card order as nvidia-smi
if ($Kv -eq "f16") { $env:Q27_KV = "f16" } else { Remove-Item Env:Q27_KV -ErrorAction SilentlyContinue }

$Exe = ".\build\q27_server.exe"
foreach ($f in @($Exe, $Model)) {
  if (-not (Test-Path $f)) { Write-Host "MISSING: $f" -ForegroundColor Red; exit 1 }
}

# Another server would hold the VRAM. This script does not stop it.
$other = Get-Process llama-server, q27_server -ErrorAction SilentlyContinue
if ($other) {
  $other | ForEach-Object { Write-Host "already running: $($_.ProcessName) (PID $($_.Id))" -ForegroundColor Red }
  Write-Host "Stop it first, for example:  Get-Process llama-server | Stop-Process" -ForegroundColor Yellow
  exit 1
}

$cards = @(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader)
$cards | ForEach-Object { Write-Host "  $_" }
if ($cards.Count -lt 2) {
  Write-Host "Only $($cards.Count) card found. The engine needs both." -ForegroundColor Red
  exit 1
}

$arguments = @("-m", $Model, "--host", $Bind, "--port", "$Port", "--reasoning-effort", $Effort)
$vis = "text only"
if ($Mmproj -and (Test-Path $Mmproj)) {
  # images: encoder on card 1; small images are scaled up to at least 1024 tokens, as production does
  $arguments += @("--mmproj", $Mmproj, "--image-min-tokens", "1024")
  $vis = "vision on"
}
if ($Ctx -gt 0) { $arguments += @("--ctx", "$Ctx") }
if ($LogDir) { $arguments += @("--log-dir", $LogDir) }
if ($ApiKey) { $env:Q27_API_KEY = $ApiKey }

Write-Host "starting q27_server on http://${Bind}:$Port (KV $Kv, effort $Effort, $vis)" -ForegroundColor Cyan
& $Exe @arguments
