@echo off
setlocal

set "MSVC_VER=14.44.35207"
set "WINSDK_VER=10.0.26100.0"
set "MSVC_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\%MSVC_VER%"
set "WINSDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "CL_EXE=%MSVC_ROOT%\bin\Hostx64\x64\cl.exe"
set "SOURCE_FILE=%~1"
set "OUTPUT_FILE=%~dp0..\hss_sistem.exe"

if "%SOURCE_FILE%"=="" set "SOURCE_FILE=%~dp0..\main.cpp"

if exist "%CL_EXE%" goto build
echo HATA: cl.exe bulunamadi: %CL_EXE%
exit /b 1

:build

set "PATH=%MSVC_ROOT%\bin\Hostx64\x64;C:\opencv\build\x64\vc16\bin;%PATH%"

"%CL_EXE%" /EHsc /Zi /std:c++17 ^
 "%SOURCE_FILE%" ^
 /I "%MSVC_ROOT%\include" ^
 /I "%WINSDK_ROOT%\Include\%WINSDK_VER%\ucrt" ^
 /I "%WINSDK_ROOT%\Include\%WINSDK_VER%\um" ^
 /I "%WINSDK_ROOT%\Include\%WINSDK_VER%\shared" ^
 /I "%WINSDK_ROOT%\Include\%WINSDK_VER%\winrt" ^
 /I "%WINSDK_ROOT%\Include\%WINSDK_VER%\cppwinrt" ^
 /I "C:\opencv\build\include" ^
 /link ^
 /LIBPATH:"%MSVC_ROOT%\lib\x64" ^
 /LIBPATH:"%WINSDK_ROOT%\Lib\%WINSDK_VER%\ucrt\x64" ^
 /LIBPATH:"%WINSDK_ROOT%\Lib\%WINSDK_VER%\um\x64" ^
 /LIBPATH:"C:\opencv\build\x64\vc16\lib" ^
 opencv_world4120.lib ^
 /out:"%OUTPUT_FILE%"
exit /b %errorlevel%
