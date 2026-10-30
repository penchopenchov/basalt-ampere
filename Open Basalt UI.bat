@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Follow the Windows setup in docs\ampere.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8088/health',timeout=2)).get('service')=='basalt'" >nul 2>&1
if errorlevel 1 start "Basalt Ampere" "%ComSpec%" /d /c call "%~dp0Start Basalt.bat"
set "BASALT_WAIT=0"
:wait_ui
".venv\Scripts\python.exe" -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8088/health',timeout=2)).get('service')=='basalt'" >nul 2>&1
if not errorlevel 1 goto open_ui
set /a BASALT_WAIT+=1 >nul
if %BASALT_WAIT% GEQ 30 goto failed
".venv\Scripts\python.exe" -c "import time; time.sleep(1)"
goto wait_ui
:open_ui
start "" "http://127.0.0.1:8088/"
exit /b 0
:failed
echo Basalt did not start at http://127.0.0.1:8088/.
echo Check the Basalt server window for an error.
pause
exit /b 1
