@echo off
rem Build environment for every .bat in this repo: MSVC (Visual Studio 2022, x64) and CUDA 13.4.
rem   Q27_VCVARS  path of vcvarsall.bat   (default: the first Visual Studio 2022 edition found)
rem   Q27_CUDA    CUDA 13.4 folder        (default: %USERPROFILE%\cuda\v13.4, else the standard install folder)
rem CUDA 13.2 miscompiles IQ3_S on sm_120 (llama.cpp PR #27902); CMakeLists.txt refuses anything older than 13.4.
rem Reference tools (bench\build-llama-*.bat) also read:
rem   Q27_LLAMA_BIN  folder with a llama.cpp build (llama.dll, llama-common.dll, mtmd.dll, ...)
rem   Q27_LLAMA_SRC  the llama.cpp source tree of the same commit (headers)
set "Q27_PF86=%ProgramFiles(x86)%"
if defined Q27_VCVARS goto have_vs
for %%e in (BuildTools Community Professional Enterprise) do if not defined Q27_VCVARS if exist "%Q27_PF86%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvarsall.bat" set "Q27_VCVARS=%Q27_PF86%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvarsall.bat"
for %%e in (BuildTools Community Professional Enterprise) do if not defined Q27_VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvarsall.bat" set "Q27_VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvarsall.bat"
:have_vs
if not defined Q27_VCVARS (
  echo Visual Studio 2022 not found. Set Q27_VCVARS to its vcvarsall.bat.
  exit /b 1
)
call "%Q27_VCVARS%" x64 >nul 2>nul || exit /b 1

if not defined Q27_CUDA if exist "%USERPROFILE%\cuda\v13.4\bin\nvcc.exe" set "Q27_CUDA=%USERPROFILE%\cuda\v13.4"
if not defined Q27_CUDA set "Q27_CUDA=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not exist "%Q27_CUDA%\bin\nvcc.exe" (
  echo CUDA 13.4 not found in "%Q27_CUDA%". Set Q27_CUDA to the CUDA 13.4 folder.
  exit /b 1
)
set "CUDA_PATH=%Q27_CUDA%"
set "CUDAToolkit_ROOT=%Q27_CUDA%"
set "PATH=%Q27_CUDA%\bin;%Q27_CUDA%\bin\x64;%PATH%"

if not defined Q27_LLAMA_BIN set "Q27_LLAMA_BIN=%~dp0..\..\qwen38_27\bin-parches"
if not defined Q27_LLAMA_SRC set "Q27_LLAMA_SRC=%~dp0..\..\llama-rig2"
exit /b 0
