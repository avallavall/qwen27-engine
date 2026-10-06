# Starts the q27 engine in place of a llama-server that runs on port 8080 (drop-in replacement): same port, the API
# key that Qwen Code sends, vision on, reasoning effort medium. Qwen Code (and other clients) need no config change.
# Usage:  powershell -File arranca-q27.ps1            ->  http://127.0.0.1:8080
#         powershell -File arranca-q27.ps1 -Port 8081  (test port, next to a running llama-server is not possible:
#                                                       both need the VRAM)
# The API key is read when the script runs: -ApiKey, else $env:Q27_API_KEY, else QWEN_LOCAL_API_KEY from
# ~\.qwen\.env (the key Qwen Code sends). It is never written to a file.
param(
  [int]$Port = 8080,
  [string]$Bind = "0.0.0.0",     # 0.0.0.0 = reachable over LAN and VPN, as production
  [string]$ApiKey = "",
  [string]$Effort = "medium",    # default reasoning effort: low, medium, xhigh (high is taken as xhigh)
  [int]$Ctx = 0,                 # 0 = the largest context that fits (262144 with q8_0 KV)
  [string]$Kv = "q8_0",          # q8_0 or f16
  [string]$Model = "",           # model files: see Find-ModelFile below
  [string]$Mmproj = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"   # same card order as nvidia-smi
if ($Kv -eq "f16") { $env:Q27_KV = "f16" } else { Remove-Item Env:Q27_KV -ErrorAction SilentlyContinue }

$Exe    = ".\build\q27_server.exe"
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
$Modelo = Find-ModelFile $Model "Q27_MODEL" "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
$Mmproj = Find-ModelFile $Mmproj "Q27_MMPROJ" "mmproj-Qwen3.8-27B-BF16.gguf"
$Log    = ".\arranque-q27.log"
$Url    = "http://127.0.0.1:$Port"

foreach ($f in @($Exe, $Modelo)) {
  if (-not (Test-Path $f)) { Write-Host "MISSING: $f" -ForegroundColor Red; exit 1 }
}

# ---- API key (never printed, never saved)
if (-not $ApiKey) { $ApiKey = $env:Q27_API_KEY }
$fuente = "-ApiKey / Q27_API_KEY"
if (-not $ApiKey) {
  $envFile = Join-Path $env:USERPROFILE ".qwen\.env"
  if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*QWEN_LOCAL_API_KEY\s*=' } | Select-Object -First 1
    if ($line) { $ApiKey = ($line -replace '^\s*QWEN_LOCAL_API_KEY\s*=\s*', '').Trim().Trim('"').Trim("'") }
  }
  $fuente = "~\.qwen\.env (QWEN_LOCAL_API_KEY)"
}
if (-not $ApiKey) {
  Write-Host "No API key found (-ApiKey, Q27_API_KEY or ~\.qwen\.env). The server would be open; stopping." -ForegroundColor Red
  exit 1
}

# ---- another server holds the VRAM (both need both cards)
$otros = @(Get-Process llama-server, q27_server -ErrorAction SilentlyContinue)
if ($otros.Count -gt 0) {
  $otros | ForEach-Object { Write-Host "running: $($_.ProcessName) (PID $($_.Id))" -ForegroundColor Yellow }
  $r = Read-Host "Stop it and start q27 instead? [y/N]"
  if ($r -notmatch '^(y|yes|s|si)$') { Write-Host "Nothing changed." ; exit 1 }
  $otros | ForEach-Object { Stop-Process -Id $_.Id -Force }
  Start-Sleep -Seconds 3
}

$tarjetas = @(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader)
$tarjetas | ForEach-Object { Write-Host "  $_" }
if ($tarjetas.Count -lt 2) {
  Write-Host "Only $($tarjetas.Count) card found. The model needs both." -ForegroundColor Red
  Write-Host "Turn on BOTH docks BEFORE the PC (OcuLink is not hot-plug)." -ForegroundColor Yellow
  exit 1
}

$argumentos = @("-m", $Modelo, "--host", $Bind, "--port", "$Port", "--reasoning-effort", $Effort,
                "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0")
$vis = "text only"
if (Test-Path $Mmproj) {
  $argumentos += @("--mmproj", $Mmproj, "--image-min-tokens", "1024")   # encoder on card 1
  $vis = "vision ON"
}
if ($Ctx -gt 0) { $argumentos += @("--ctx", "$Ctx") }

$env:Q27_API_KEY = $ApiKey   # the server reads the key from the environment, not from the command line
Write-Host "`nstarting q27 on $Url (KV $Kv, effort $Effort, $vis; key from $fuente; log: $Log)" -ForegroundColor Cyan
$srv = Start-Process -FilePath $Exe -ArgumentList $argumentos -PassThru -NoNewWindow `
        -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

# ---- wait for /health (about 10 s when the model file is in the OS cache, 1-2 min cold)
$listo = $false
for ($i = 0; $i -lt 300; $i++) {
  try { if ((Invoke-RestMethod "$Url/health" -TimeoutSec 2).status -eq "ok") { $listo = $true; break } } catch {}
  if ($srv.HasExited) { break }
  Start-Sleep -Seconds 2
}
if (-not $listo) {
  Write-Host "`nDID NOT START. Last lines:" -ForegroundColor Red
  Get-Content "$Log.err" -Tail 15 -ErrorAction SilentlyContinue
  exit 1
}
Get-Content "$Log.err" -Tail 1

$cab = @{ "Authorization" = "Bearer $ApiKey" }
try {
  $p = Invoke-RestMethod "$Url/props" -Headers $cab -TimeoutSec 10
  Write-Host ("  context {0} tokens, KV {1}, vision {2}" -f $p.default_generation_settings.n_ctx, $p.kv_cache_type, $p.modalities.vision) -ForegroundColor Green
} catch {}
$usadas = @(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | ForEach-Object { [int]$_ }
Write-Host "  VRAM: $($usadas -join ' + ') MiB" -ForegroundColor Green

# ---- one real request
try {
  $cuerpo = @{ messages = @(@{ role = "user"; content = "Say OK." }); max_tokens = 200 } | ConvertTo-Json -Depth 5
  $r = Invoke-RestMethod "$Url/v1/chat/completions" -Method Post -Body $cuerpo -ContentType "application/json" -Headers $cab -TimeoutSec 180
  Write-Host "  OK: it generates ($([math]::Round($r.timings.predicted_per_second,1)) tok/s cold): $($r.choices[0].message.content.Trim())" -ForegroundColor Green
} catch {
  Write-Host "  FAIL: it loaded but did not generate." -ForegroundColor Red
  Get-Content "$Log.err" -Tail 6 -ErrorAction SilentlyContinue
}

Write-Host "`n  READY  ->  $Url" -ForegroundColor Green
Write-Host "  stop   ->  Get-Process q27_server | Stop-Process -Force" -ForegroundColor DarkGray
Write-Host "  back to llama.cpp -> stop q27, then start llama-server again`n" -ForegroundColor DarkGray
Wait-Process -Id $srv.Id
