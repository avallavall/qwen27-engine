@echo off
call "%~dp0..\..\tools\env.bat" || exit /b 1
cd /d "%~dp0"
set "BIN=%Q27_LLAMA_BIN%"
set "SRC=%Q27_LLAMA_SRC%"
set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist llama-common.lib (
  dumpbin /nologo /exports "%BIN%\llama-common.dll" > common.exports.txt || exit /b 1
  %PY% mkdef.py common.exports.txt llama-common.dll llama-common.def || exit /b 1
  lib /nologo /def:llama-common.def /machine:x64 /out:llama-common.lib >nul || exit /b 1
)
cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /Zc:__cplusplus /I"%SRC%\common" /I"%SRC%\include" /I"%SRC%\ggml\include" /I"%SRC%\vendor" oracle.cpp llama-common.lib /Fe:oracle.exe || exit /b 1
