@echo off
title SpectraTrack AI - Tactical Detection & Tracking Dashboard
echo ========================================================
echo   Starting SpectraTrack AI (Web Dashboard Mode)
echo ========================================================
cd /d "%~dp0"

echo Memeriksa dan membersihkan port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo Menyiapkan model AI dan menyalakan server...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"
.\.v\Scripts\python.exe web_app.py
pause
