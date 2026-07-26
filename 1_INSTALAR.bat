@echo off
setlocal
cd /d "%~dp0"

echo Criando o ambiente virtual...
where py >nul 2>nul
if errorlevel 1 goto python_error

py -3.13 -m venv .venv
if errorlevel 1 py -m venv .venv
if not exist ".venv\Scripts\python.exe" goto python_error

echo Instalando as bibliotecas...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error

echo.
echo Instalacao concluida.
echo Agora execute 2_TESTAR.bat.
pause
exit /b 0

:python_error
echo.
echo Python nao encontrado ou ambiente virtual nao criado.
echo Instale o Python 3.13 em https://www.python.org/downloads/
pause
exit /b 1

:install_error
echo.
echo Ocorreu um erro durante a instalacao.
pause
exit /b 1
