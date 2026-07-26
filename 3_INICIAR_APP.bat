@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto not_installed

".venv\Scripts\python.exe" -m streamlit run app.py
exit /b %errorlevel%

:not_installed
echo Execute primeiro o arquivo 1_INSTALAR.bat.
pause
exit /b 1
