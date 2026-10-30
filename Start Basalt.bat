@echo off
setlocal
cd /d "%~dp0"
if not exist "bin\basalt.exe" (
    echo Missing bin\basalt.exe. Build Basalt before starting it.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Follow the Windows setup in docs\ampere.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8088/health',timeout=2)).get('service')=='basalt'" >nul 2>&1
if not errorlevel 1 (
    echo Basalt is already running at http://127.0.0.1:8088/
    exit /b 0
)
title Basalt Ampere
echo Starting Basalt. Keep this window open while using it.
echo Interface: http://127.0.0.1:8088/
"bin\basalt.exe" serve --config "basalt_config.json"
set "BASALT_EXIT=%ERRORLEVEL%"
if not "%BASALT_EXIT%"=="0" (
    echo Basalt stopped with exit code %BASALT_EXIT%.
    pause
)
exit /b %BASALT_EXIT%
