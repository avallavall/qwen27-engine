@echo off
rem Build the engine on Windows: Visual Studio 2022 (MSVC) + CUDA 13.4 + CMake + Ninja. Output in build\.
rem tools\env.bat finds the compilers (or set Q27_VCVARS / Q27_CUDA). Extra arguments go to "cmake --build",
rem e.g. "build.bat --target q27_server".
call "%~dp0tools\env.bat" || exit /b 1
cd /d "%~dp0"
if not exist build\build.ninja (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120a-real ^
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl || exit /b 1
)
cmake --build build %* || exit /b 1
