@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Live German - English Captions
".venv\Scripts\python.exe" live_captions_translate.py
echo.
echo (captions stopped) - press any key to close
pause >nul
