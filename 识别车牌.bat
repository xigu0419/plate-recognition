@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Users\19500\.conda\envs\plate\python.exe" "%~dp0plate_recognition.py" %*
pause