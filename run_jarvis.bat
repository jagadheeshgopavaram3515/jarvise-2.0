@echo off
REM Launch the local FastAPI backend and Next.js interface.
cd /d "%~dp0"
python run_web.py
if errorlevel 1 pause
