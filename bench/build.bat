@echo off
rem Build the micro-benchmarks with CUDA 13.4 (never 13.2) and MSVC 14.44.
call "%~dp0..\tools\env.bat" || exit /b 1
if not exist "%~dp0build" mkdir "%~dp0build"
nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -o "%~dp0build\hw.exe" "%~dp0hw.cu" || exit /b 1
copy /y "%CUDA_PATH%\bin\x64\cudart64_13.dll" "%~dp0build\" >nul 2>&1
echo built %~dp0build\hw.exe
