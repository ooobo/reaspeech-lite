@echo off
REM Build standalone executable for ReaSpeech Parakeet transcription (Windows)

echo Building Parakeet Transcription Executable for Windows
echo ======================================================

REM Check if we're in the right directory
if not exist parakeet_transcribe.py (
    echo Error: Must run from python\ directory
    exit /b 1
)

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Install dependencies
echo Installing Python dependencies...
pip install onnx-asr

REM Build executable
echo Running PyInstaller...
pyinstaller ^
    --onefile ^
    --name parakeet-transcribe-windows ^
    --clean ^
    --noconfirm ^
    parakeet_transcribe.py

echo.
echo Build complete!
echo Executable location: dist\parakeet-transcribe-windows.exe
echo.
echo To test:
echo   dist\parakeet-transcribe-windows.exe C:\path\to\audio.wav
