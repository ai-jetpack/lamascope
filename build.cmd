@echo off
rem Builds LlamaScope (Release) with the newest Visual Studio that has the C++ x64 tools.
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo Visual Studio with C++ tools not found.
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"
cmake -S "%~dp0." -B "%~dp0build\%CFG%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% || exit /b 1
cmake --build "%~dp0build\%CFG%" || exit /b 1
echo.
echo Built: %~dp0build\%CFG%\LlamaScope.exe
