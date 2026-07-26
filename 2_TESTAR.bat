@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto not_installed

".venv\Scripts\python.exe" verificar_projeto.py
if errorlevel 1 goto test_error

echo.
echo Teste concluido com sucesso.
pause
exit /b 0

:not_installed
echo Execute primeiro o arquivo 1_INSTALAR.bat.
pause
exit /b 1

:test_error
echo.
echo O teste encontrou um erro.
pause
exit /b 1
