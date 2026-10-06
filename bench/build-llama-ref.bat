@echo off
rem Build bench\build\llama_ref.exe against the production llama.dll (bin-parches, read only).
rem An import library is generated from the DLL exports; headers come from llama-rig2 (same commit).
call "%~dp0..\tools\env.bat" || exit /b 1
cd /d "%~dp0"
set "BIN=%Q27_LLAMA_BIN%"
set "SRC=%Q27_LLAMA_SRC%"
if not exist build mkdir build
dumpbin /nologo /exports "%BIN%\llama.dll" > build\llama.exports.txt || exit /b 1
echo LIBRARY llama.dll> build\llama.def
echo EXPORTS>> build\llama.def
for /f "skip=19 tokens=4" %%a in (build\llama.exports.txt) do echo %%a>> build\llama.def
lib /nologo /def:build\llama.def /machine:x64 /out:build\llama.lib >nul || exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /utf-8 /I"%SRC%\include" /I"%SRC%\ggml\include" llama_ref.cpp build\llama.lib /Fe:build\llama_ref.exe /Fo:build\ || exit /b 1
