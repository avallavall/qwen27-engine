@echo off
rem Build bench\build\llama_vision.exe (vision oracle) against the production mtmd.dll, llama.dll and ggml-base.dll
rem (qwen38_27\bin-parches, read only). Import libraries are generated from the DLL exports if missing (only the
rem plain C names for mtmd and ggml-base); headers come from llama-rig2 (same commit as the DLLs).
call "%~dp0..\tools\env.bat" || exit /b 1
cd /d "%~dp0"
set "BIN=%Q27_LLAMA_BIN%"
set "SRC=%Q27_LLAMA_SRC%"
if not exist build mkdir build
if not exist build\llama.lib (
  dumpbin /nologo /exports "%BIN%\llama.dll" > build\llama.exports.txt || exit /b 1
  echo LIBRARY llama.dll> build\llama.def
  echo EXPORTS>> build\llama.def
  for /f "skip=19 tokens=4" %%a in (build\llama.exports.txt) do echo %%a>> build\llama.def
  lib /nologo /def:build\llama.def /machine:x64 /out:build\llama.lib >nul || exit /b 1
)
if not exist build\mtmd.lib (
  dumpbin /nologo /exports "%BIN%\mtmd.dll" > build\mtmd.exports.txt || exit /b 1
  echo LIBRARY mtmd.dll> build\mtmd.def
  echo EXPORTS>> build\mtmd.def
  for /f "tokens=4" %%a in ('findstr /c:" mtmd_" build\mtmd.exports.txt') do echo %%a>> build\mtmd.def
  lib /nologo /def:build\mtmd.def /machine:x64 /out:build\mtmd.lib >nul || exit /b 1
)
if not exist build\ggml-base.lib (
  dumpbin /nologo /exports "%BIN%\ggml-base.dll" > build\ggml-base.exports.txt || exit /b 1
  echo LIBRARY ggml-base.dll> build\ggml-base.def
  echo EXPORTS>> build\ggml-base.def
  for /f "tokens=4" %%a in ('findstr /c:" ggml_" /c:" gguf_" build\ggml-base.exports.txt') do echo %%a>> build\ggml-base.def
  lib /nologo /def:build\ggml-base.def /machine:x64 /out:build\ggml-base.lib >nul || exit /b 1
)
cl /nologo /O2 /EHs /std:c++17 /utf-8 /I"%SRC%\include" /I"%SRC%\ggml\include" /I"%SRC%\tools\mtmd" llama_vision.cpp build\mtmd.lib build\llama.lib build\ggml-base.lib /Fe:build\llama_vision.exe /Fo:build\ || exit /b 1
