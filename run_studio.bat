@echo off
setlocal enabledelayedexpansion

title RoadSentinel Studio v3.0 - Autonomous Perception Platform
color 0A

echo ===============================================================================
echo                ROADSENTINEL STUDIO v3.0 // MISSION CONTROL
echo        Autonomous VRU Safety Platform ^| Active Foveation ^| AWS COOL
echo ===============================================================================
echo.

:: Check Python Installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ and add it to your PATH environment variable.
    pause
    exit /b 1
)

:: Display Python Version
for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo [INFO] Detected: %PY_VER%

:: Check if requirements are satisfied
echo [INFO] Verifying core platform dependencies...
python -c "import fastapi, uvicorn, torch, ultralytics, cv2" >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Missing required dependencies. Installing from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install requirements. Please check your network connection.
        pause
        exit /b 1
    )
)

:: Launch Browser in Background after 2 seconds
start "" powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:8000'"

:: Start the Studio Application Server
echo.
echo ===============================================================================
echo [STATUS] Launching RoadSentinel Studio Server at http://localhost:8000 ...
echo [CTRL+C] Press Ctrl+C in this terminal window to stop the server.
echo ===============================================================================
echo.

python saas_app.py

pause
