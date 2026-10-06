@echo off
call "%~dp0..\..\tools\env.bat" || exit /b 1
cd /d "%~dp0"
set "REPO=%~dp0..\.."
cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /Zc:__cplusplus /DNDEBUG /I"%REPO%\src" /I"%REPO%\third_party" runner.cpp "%REPO%\build-parse\q27text.lib" /Fe:runner.exe || exit /b 1
