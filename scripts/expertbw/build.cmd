@echo off
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /arch:AVX2 /std:c++20 /EHsc expertbw.cpp advapi32.lib /Fe:expertbw.exe || exit /b 1
echo BUILD OK
