@echo off
setlocal

set "APP=%~1"
if "%APP%"=="" set "APP=%~dp0..\hss_sistem.exe"

if exist "C:\Users\askan\AppData\Local\Programs\Python\Python312\python.exe" (
    set "KORGAN_PYTHON=C:\Users\askan\AppData\Local\Programs\Python\Python312\python.exe"
)

set "PATH=C:\opencv\build\x64\vc16\bin;%PATH%"
"%APP%"
exit /b %errorlevel%
