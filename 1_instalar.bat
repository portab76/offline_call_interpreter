@echo off
rem Instala la conversacion traducida en este ordenador: crea el entorno de
rem Python (carpeta venv) con los paquetes necesarios. No descarga modelos:
rem 2_conversacion.bat baja a la carpeta "modelos" los que necesita la primera vez
rem (hace falta internet esa vez; despues funciona sin conexion).
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
echo === Instalando el traductor ===
where py >nul 2>nul
if errorlevel 1 (
  echo.
  echo No encuentro Python.
  where winget >nul 2>nul
  if errorlevel 1 goto :sin_python
  choice /m "Quieres instalar Python 3.12 ahora con winget"
  if errorlevel 2 goto :sin_python
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  where py >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Python instalado. Cierra esta ventana y vuelve a abrir 1_instalar.bat.
    pause
    exit /b 1
  )
)
rem El entorno (venv) se reutiliza si funciona; si no (p. ej. se desinstalo
rem o actualizo el Python con el que se creo), se borra y se vuelve a crear.
if exist venv\Scripts\python.exe (
  venv\Scripts\python -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo El entorno de Python ya no funciona: se vuelve a crear.
    rmdir /s /q venv
  ) else (
    echo Ya hay un entorno en venv: se actualizan sus paquetes.
  )
)
if not exist venv\Scripts\python.exe (
  rem Se usa el primer Python que arranque de verdad (py -3 puede apuntar a
  rem una version registrada pero borrada).
  set "PYV="
  for %%v in (3.12 3.11 3.13 3.14 3) do if not defined PYV (
    py -%%v -c "import sys" >nul 2>nul && set "PYV=%%v"
  )
  if not defined PYV goto :sin_python
  echo Creando el entorno con Python !PYV! ...
  py -!PYV! -m venv venv
  if errorlevel 1 ( echo Error creando el entorno de Python. & pause & exit /b 1 )
)
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Error instalando paquetes. & pause & exit /b 1 )
echo.
echo Instalacion terminada. Abre 2_conversacion.bat.
echo La primera vez descargara los modelos que necesite (unos 500 MB para
echo espanol-ingles; cada idioma nuevo, unos 200 MB mas): tarda unos minutos.
pause
exit /b 0

:sin_python
echo Instalalo con este comando en una ventana de PowerShell:
echo     winget install Python.Python.3.12
echo o descargalo de https://www.python.org/downloads/ ^(marca "Add python.exe to PATH"^).
echo Luego vuelve a ejecutar este archivo.
pause
exit /b 1
