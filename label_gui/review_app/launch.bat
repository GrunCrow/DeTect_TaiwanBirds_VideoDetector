@echo off
REM Launch script for Label Review App

echo.
echo ========================================
echo  Label Review Web App Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ERROR: app.py not found
    echo Please run this script from the review_app directory
    pause
    exit /b 1
)

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Make sure you have pip installed
    pause
    exit /b 1
)

REM Start the app
echo.
echo Starting Label Review App...
echo.
echo Dashboard: http://localhost:5000/
echo Review:    http://localhost:5000/review
echo.
echo Press CTRL+C to stop the server
echo.

python app.py

pause
