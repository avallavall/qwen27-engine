@echo off
rem Start the production llama.cpp build under Nsight Systems. Args: <ctx> <duration_s> <report_name> [extra nsys option, e.g. --delay=240].
rem Same flags as the production start script, except: port 8081, 127.0.0.1, no API key. Output in bench\out.
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set LLAMA_SCHED_POOL=8
set LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_effort":"medium"}
if not defined Q27_PROD set "Q27_PROD=%~dp0..\..\qwen38_27"
set "P=%Q27_PROD%"
set NSYS="C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
cd /d "%~dp0out"
%NSYS% profile --duration=%2 %4 --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite=true -o %3 ^
  "%P%\bin-parches\llama-server.exe" -m "%P%\models\Qwen3.8-27B\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" ^
  --mmproj "%P%\models\Qwen3.8-27B\mmproj-Qwen3.8-27B-BF16.gguf" --image-min-tokens 1024 -mmdev CUDA1 ^
  -c %1 --load-mode none -fa auto -cram 8192 -ctxcp 4 -np 1 -ngl 66 -sm tensor -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-sampling probabilistic -ub 1024 -b 2048 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --host 127.0.0.1 --port 8081 --log-file llama-app.log > llama.log 2> llama.log.err
