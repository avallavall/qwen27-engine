@echo off
rem Build bench\build\llama_chat.exe (chat template + output parser oracle) against the production
rem llama-common.dll and ggml-base.dll (qwen38_27\bin-parches, read only).
rem Import libraries are generated from the DLL exports if missing (build\chat\); headers come from llama-rig2
rem (same commit as the DLLs). Compiler settings match the DLLs: /MD (dynamic CRT), release, C++17, /utf-8,
rem so std::string, std::vector and std::map cross the DLL boundary safely.
rem Run: set PATH=%Q27_LLAMA_BIN%;%PATH%
rem      bench\build\llama_chat.exe <log dir> [--gguf PATH] [--show N]
call "%~dp0..\tools\env.bat" || exit /b 1
cd /d "%~dp0"
set "BIN=%Q27_LLAMA_BIN%"
set "SRC=%Q27_LLAMA_SRC%"
if not exist build\chat mkdir build\chat
if not exist build\chat\llama-common.lib (
  dumpbin /nologo /exports "%BIN%\llama-common.dll" > build\chat\llama-common.exports.txt || exit /b 1
  rem C++ exports only (mangled names start with ?), without the thread-safe-static guards (?$TSS0..., data).
  rem The DLL also exports C runtime helpers (printf, _Avx2WmemEnabledWeakValue, ...); those must not be imported.
  findstr /l /c:" ?" build\chat\llama-common.exports.txt | findstr /l /v /c:" ?$TSS0" > build\chat\llama-common.cxx.txt
  >build\chat\llama-common.def echo LIBRARY llama-common.dll
  >>build\chat\llama-common.def echo EXPORTS
  for /f "tokens=4" %%a in (build\chat\llama-common.cxx.txt) do >>build\chat\llama-common.def echo %%a
  lib /nologo /def:build\chat\llama-common.def /machine:x64 /out:build\chat\llama-common.lib >nul || exit /b 1
)
if not exist build\chat\ggml-base.lib (
  dumpbin /nologo /exports "%BIN%\ggml-base.dll" > build\chat\ggml-base.exports.txt || exit /b 1
  >build\chat\ggml-base.def echo LIBRARY ggml-base.dll
  >>build\chat\ggml-base.def echo EXPORTS
  for /f "tokens=4" %%a in ('findstr /l /c:" gguf_" build\chat\ggml-base.exports.txt') do >>build\chat\ggml-base.def echo %%a
  lib /nologo /def:build\chat\ggml-base.def /machine:x64 /out:build\chat\ggml-base.lib >nul || exit /b 1
)
cl /nologo /O2 /MD /DNDEBUG /EHsc /std:c++17 /utf-8 /W3 /wd4996 /I"%SRC%\common" /I"%SRC%\include" /I"%SRC%\ggml\include" ^
  llama_chat.cpp build\chat\llama-common.lib build\chat\ggml-base.lib /Fe:build\llama_chat.exe /Fo:build\chat\ || exit /b 1
