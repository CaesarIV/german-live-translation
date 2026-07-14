@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Live Caption Overlay
".venv\Scripts\python.exe" overlay.py
echo.
echo (overlay closed) - press any key to close
pause >nul
