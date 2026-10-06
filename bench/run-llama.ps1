# Start the production llama.cpp build (qwen38_27\bin-parches) with the production flags, for measuring.
# Differences from qwen38_27\arranca.ps1: port 8081, 127.0.0.1 only, no API key, logs in bench\out.
# Nothing is written into qwen38_27.
# Usage: powershell -File bench\run-llama.ps1 [-Ctx 180224] [-NoVision] [-Nsys]
#   -Nsys: start under Nsight Systems in an interactive session named "q27" (collection starts paused).
param([int] $Ctx = 180224, [switch] $NoVision, [switch] $Nsys)

$ErrorActionPreference = "Stop"
# Q27_PROD: folder with bin-parches\ (llama.cpp build) and models\Qwen3.8-27B\ (default: ..\qwen38_27 next to this repo)
$Prod  = if ($env:Q27_PROD) { $env:Q27_PROD } else { Join-Path $PSScriptRoot "..\..\qwen38_27" }
$Out   = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force $Out | Out-Null
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:LLAMA_SCHED_POOL  = "8"

$a = @("-m", "$Prod\models\Qwen3.8-27B\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
if (-not $NoVision) {
  $a += @("--mmproj", "$Prod\models\Qwen3.8-27B\mmproj-Qwen3.8-27B-BF16.gguf", "--image-min-tokens", "1024", "-mmdev", "CUDA1")
}
$a += @("-c", "$Ctx", "--load-mode", "none", "-fa", "auto", "-cram", "8192", "-ctxcp", "4", "-np", "1",
        "-ngl", "66", "-sm", "tensor", "-ctk", "f16", "-ctv", "f16", "-ctkd", "f16", "-ctvd", "f16",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-sampling", "probabilistic",
        "-ub", "1024", "-b", "2048", "--chat-template-kwargs", '{\"reasoning_effort\":\"medium\"}',
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
        "--host", "127.0.0.1", "--port", "8081")

$exe = "$Prod\bin-parches\llama-server.exe"
if ($Nsys) {
  $NsysExe = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
  $na = @("profile", "--session-new=q27", "--start-later=true", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--force-overwrite=true", "-o", (Join-Path $Out "q27")) + @($exe) + $a
  $p = Start-Process -FilePath $NsysExe -ArgumentList $na -PassThru -NoNewWindow -WorkingDirectory $Out `
         -RedirectStandardOutput "$Out\llama.log" -RedirectStandardError "$Out\llama.log.err"
} else {
  $p = Start-Process -FilePath $exe -ArgumentList $a -PassThru -NoNewWindow -WorkingDirectory $Out `
         -RedirectStandardOutput "$Out\llama.log" -RedirectStandardError "$Out\llama.log.err"
}
for ($i = 0; $i -lt 300; $i++) {
  try { if ((Invoke-RestMethod "http://127.0.0.1:8081/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
  if (-not $Nsys -and $p.HasExited) { Write-Host "exited"; Get-Content "$Out\llama.log.err" -Tail 20; exit 1 }
  Start-Sleep -Seconds 2
}
$used = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
Write-Host "ready pid=$($p.Id) vram_used_mib=$($used -join ',')"
