@echo off
rem Conversacion traducida en los dos sentidos: lo que dice el otro se lee
rem traducido y lo que digo yo se oye traducido (voz de Piper).
rem
rem Idiomas: es, en, fr, de, it, ca, pt, ru, zh, vi, ar (y los que se anadan en
rem Herramientas - Idiomas), en cualquier combinacion. Sin argumentos se usan los
rem ultimos elegidos.
rem   2_conversacion.bat --yo es --el fr
rem   2_conversacion.bat --yo de --el it --modo presencial
rem   (--modo llamada: el suena por el PC; presencial: esta a mi lado)
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe ( echo Primero ejecuta 1_instalar.bat & pause & exit /b 1 )
venv\Scripts\python conversacion.py %*
if errorlevel 1 pause
