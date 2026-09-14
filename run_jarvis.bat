@echo off
REM Always launches Jarvis with the Python that has all the packages installed.
REM Double-click this file to run the assistant.
cd /d "D:\JARVIS"
 d:\GenAi_Project\medibot\medibot-env\Scripts\Activate.ps1
>> python main.py
echo.
echo Jarvis has stopped. Press any key to close.
pause >nul
