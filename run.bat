@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
".venv\Scripts\python.exe" Cropyield.py
pause
