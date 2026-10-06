@echo off
rem Build one target in its own build folder (so parallel work does not share build\).
rem Usage: tools\build_target.bat <build_dir> <target>
call "%~dp0env.bat" || exit /b 1
cd /d "%~dp0\.."
if not exist %1\build.ninja (
  cmake -S . -B %1 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120a-real ^
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl >nul || exit /b 1
)
cmake --build %1 --target %2 || exit /b 1
