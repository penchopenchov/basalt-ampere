@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\stop_basalt.ps1" -Root "%~dp0."
set "BASALT_EXIT=%ERRORLEVEL%"
if not "%BASALT_EXIT%"=="0" pause
exit /b %BASALT_EXIT%
