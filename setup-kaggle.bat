@echo off
REM Kaggle CLI Setup Script for Windows

echo ========================================
echo   Kaggle CLI Setup
echo ========================================
echo.

REM Step 1: Check Python
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python installed
echo.

REM Step 2: Install Kaggle CLI
echo Step 2: Installing Kaggle CLI...
pip install kaggle --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Kaggle CLI
    pause
    exit /b 1
)
echo [OK] Kaggle CLI installed
echo.

REM Step 3: Create .kaggle directory
echo Step 3: Setting up credentials directory...
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
echo [OK] Directory created: %USERPROFILE%\.kaggle
echo.

REM Step 4: Check for kaggle.json
echo Step 4: Checking for API credentials...
if exist "%USERPROFILE%\Downloads\kaggle.json" (
    echo [FOUND] kaggle.json in Downloads
    move "%USERPROFILE%\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\" >nul
    echo [OK] Moved to: %USERPROFILE%\.kaggle\
) else if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo [OK] kaggle.json already in place
) else (
    echo [ACTION NEEDED] Please download your API credentials:
    echo.
    echo   1. Go to: https://www.kaggle.com/settings/account
    echo   2. Scroll to "API" section
    echo   3. Click "Create New API Token"
    echo   4. Move kaggle.json to: %USERPROFILE%\.kaggle\
    echo.
    echo After downloading, run this script again.
    pause
    exit /b 0
)
echo.

REM Step 5: Test setup
echo Step 5: Testing Kaggle CLI...
kaggle competitions list -v >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Kaggle CLI test failed
    echo Please check your credentials at: %USERPROFILE%\.kaggle\kaggle.json
    pause
    exit /b 1
)
echo [OK] Kaggle CLI working!
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo You can now use Kaggle programmatically!
echo.
echo Quick test commands:
echo   kaggle competitions list
echo   kaggle datasets list
echo   kaggle kernels list --mine
echo.
echo Your credentials: %USERPROFILE%\.kaggle\kaggle.json
echo.
pause
