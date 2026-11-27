@echo off
cd /d "%~dp0"
set "PY_EXE=%LOCALAPPDATA%\Python\bin\python3.exe"
if not exist "%PY_EXE%" set "PY_EXE=python3"
echo Starting AI Business Intelligence Report Assistant...
echo Open http://127.0.0.1:8000 in your browser (press Ctrl+C to stop).
start "" http://127.0.0.1:8000
"%PY_EXE%" server.py
pause
