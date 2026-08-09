@echo off
chcp 65001 >nul
cd /d %~dp0
call .venv\Scripts\activate.bat

if "%1"=="" (set MODEL=cnn) else (set MODEL=%1)
python -m src.train --model %MODEL% --epochs 8
pause
