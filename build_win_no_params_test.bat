@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================
REM pyVideoTrans Windows no-params test package builder
REM
REM This script reuses the local .venv310 build environment. It does
REM not install packages. Prepare the environment once, then reuse it:
REM   C:\Users\PC\miniconda3\python.exe -m venv .venv310
REM   .\.venv310\Scripts\python.exe -m pip install -r requirements.build.txt
REM   .\.venv310\Scripts\python.exe -m pip install qwen-tts
REM
REM Output:
REM   dist\pyVideoTrans-no-params-test\pyVideoTrans.exe
REM ============================================================

set "PY=.venv310\Scripts\python.exe"
set "PYINSTALLER=.venv310\Scripts\pyinstaller.exe"
set "SPEC=pyvideotrans_no_params.spec"
set "OUT_NAME=pyVideoTrans-no-params-test"
set "OUT=dist\%OUT_NAME%"

echo [1/8] Check local build environment...
if not exist "%PY%" (
    echo [ERROR] Missing %PY%
    echo Create .venv310 with Python 3.10 first.
    goto :err
)
if not exist "%PYINSTALLER%" (
    echo [ERROR] Missing %PYINSTALLER%
    echo Install dependencies into .venv310 first.
    goto :err
)
call :check_site_package qwen_tts
if errorlevel 1 goto :err
call :check_site_package sox
if errorlevel 1 goto :err

echo [2/8] Generate no-params spec...
"%PY%" -c "from pathlib import Path; src=Path('pyvideotrans.spec').read_text(encoding='utf-8'); src=src.replace('from PyInstaller.utils.hooks import collect_all, collect_submodules\n', 'from PyInstaller.utils.hooks import collect_all, collect_submodules\nimport sys; sys.setrecursionlimit(sys.getrecursionlimit() * 5)\n') if 'sys.setrecursionlimit' not in src else src; lines=[ln for ln in src.splitlines() if 'params.json' not in ln.replace('\\\\','/')]; Path('pyvideotrans_no_params.spec').write_text('\n'.join(lines)+'\n', encoding='utf-8')"
if errorlevel 1 goto :err

echo [3/8] Clean previous no-params build output...
if exist "dist\pyVideoTrans" rmdir /s /q "dist\pyVideoTrans"
if exist "%OUT%" rmdir /s /q "%OUT%"
if exist "build\pyvideotrans_no_params" rmdir /s /q "build\pyvideotrans_no_params"

echo [4/8] Run PyInstaller...
"%PYINSTALLER%" "%SPEC%" --clean --noconfirm
if errorlevel 1 goto :err
if not exist "dist\pyVideoTrans\pyVideoTrans.exe" (
    echo [ERROR] PyInstaller did not create dist\pyVideoTrans\pyVideoTrans.exe
    goto :err
)
ren "dist\pyVideoTrans" "%OUT_NAME%"
if errorlevel 1 goto :err

echo [5/8] Create runtime folders...
mkdir "%OUT%\models"       2>nul
mkdir "%OUT%\output"       2>nul
mkdir "%OUT%\logs"         2>nul
mkdir "%OUT%\tmp"          2>nul
mkdir "%OUT%\presets_user" 2>nul
mkdir "%OUT%\tools"        2>nul
copy /y "models\下载说明.txt" "%OUT%\models\下载说明.txt" >nul 2>&1

echo [6/8] Copy external runtimes...
if not exist "tools\remove_subtitles_cli.py" (
    echo [ERROR] Missing tools\remove_subtitles_cli.py
    goto :err
)
copy /y "tools\remove_subtitles_cli.py" "%OUT%\tools\remove_subtitles_cli.py" >nul
if errorlevel 1 goto :err
call :copy_required_dir "tools\video-subtitle-remover-src" "%OUT%\tools\video-subtitle-remover-src"
if errorlevel 1 goto :err
call :copy_required_dir "tools\video-subtitle-remover-env" "%OUT%\tools\video-subtitle-remover-env"
if errorlevel 1 goto :err
call :copy_required_dir "ffmpeg" "%OUT%\ffmpeg"
if errorlevel 1 goto :err

echo [7/8] Copy resource/source mirrors and local optional packages...
call :copy_optional_dir "%OUT%\_internal\videotrans" "%OUT%\videotrans"
if errorlevel 1 goto :err
for %%D in (ui component mainwin task util recognition tts translator process configure winform) do (
    call :copy_optional_dir "videotrans\%%D" "%OUT%\videotrans\%%D"
    if errorlevel 1 goto :err
    call :copy_optional_dir "videotrans\%%D" "%OUT%\_internal\videotrans\%%D"
    if errorlevel 1 goto :err
)
call :copy_site_package qwen_tts
if errorlevel 1 goto :err
call :copy_site_package qwen_tts-0.1.1.dist-info
if errorlevel 1 goto :err
call :copy_site_package sox
if errorlevel 1 goto :err
call :copy_site_package sox-1.5.0.dist-info
if errorlevel 1 goto :err
for %%P in (gradio gradio-6.12.0.dist-info fastapi fastapi-0.136.0.dist-info starlette starlette-1.0.0.dist-info uvicorn uvicorn-0.44.0.dist-info orjson orjson-3.11.8.dist-info groovy groovy-0.1.2.dist-info hf_gradio hf_gradio-0.4.1.dist-info safehttpx safehttpx-0.1.7.dist-info semantic_version semantic_version-2.10.0.dist-info python_multipart python_multipart-0.0.26.dist-info tomlkit tomlkit-0.14.0.dist-info annotated_doc annotated_doc-0.0.4.dist-info typing_inspection typing_inspection-0.4.2.dist-info) do (
    call :copy_site_package %%P optional
    if errorlevel 1 goto :err
)

echo [8/8] Remove params.json and verify...
del /f /q "%OUT%\videotrans\params.json" 2>nul
del /f /q "%OUT%\_internal\videotrans\params.json" 2>nul
for /f "delims=" %%F in ('dir /b /s "%OUT%\params.json" 2^>nul') do (
    echo [ERROR] params.json still exists: %%F
    goto :err
)
if not exist "%OUT%\pyVideoTrans.exe" goto :verify_err
if not exist "%OUT%\videotrans\styles\style.qss" goto :verify_err
if not exist "%OUT%\ffmpeg\ffmpeg.exe" goto :verify_err
if not exist "%OUT%\tools\video-subtitle-remover-env\Scripts\python.exe" goto :verify_err
if not exist "%OUT%\_internal\qwen_tts\__init__.py" goto :verify_err
if not exist "%OUT%\_internal\sox\__init__.py" goto :verify_err

echo.
echo Done.
echo Package: %CD%\%OUT%
echo Entry:   %CD%\%OUT%\pyVideoTrans.exe
echo.
exit /b 0

:check_site_package
set "PKG=%~1"
if exist ".venv310\Lib\site-packages\%PKG%" exit /b 0
if exist ".venv310\Lib\site-packages\%PKG%.py" exit /b 0
echo [ERROR] Missing .venv310\Lib\site-packages\%PKG%
echo Install/prepare it once, then rerun this script.
exit /b 1

:copy_site_package
set "PKG=%~1"
set "OPTIONAL=%~2"
set "SRC=.venv310\Lib\site-packages\%PKG%"
set "DST=%OUT%\_internal\%PKG%"
if exist "%SRC%" (
    robocopy "%SRC%" "%DST%" /E /XD __pycache__ /XF *.pyc >nul
    if %ERRORLEVEL% GEQ 8 exit /b %ERRORLEVEL%
    exit /b 0
)
if exist "%SRC%.py" (
    copy /y "%SRC%.py" "%OUT%\_internal\%PKG%.py" >nul
    exit /b %ERRORLEVEL%
)
if /i "%OPTIONAL%"=="optional" exit /b 0
echo [ERROR] Missing site package: %PKG%
exit /b 1

:copy_required_dir
set "SRC=%~1"
set "DST=%~2"
if not exist "%SRC%" (
    echo [ERROR] Missing required directory: %SRC%
    exit /b 1
)
robocopy "%SRC%" "%DST%" /E /XD .git __pycache__ /XF *.pyc >nul
if %ERRORLEVEL% GEQ 8 exit /b %ERRORLEVEL%
exit /b 0

:copy_optional_dir
set "SRC=%~1"
set "DST=%~2"
if not exist "%SRC%" exit /b 0
robocopy "%SRC%" "%DST%" /E /XD .git __pycache__ /XF params.json *.pyc >nul
if %ERRORLEVEL% GEQ 8 exit /b %ERRORLEVEL%
exit /b 0

:verify_err
echo [ERROR] Package verification failed. Required runtime file is missing.
goto :err

:err
echo.
echo *** Build failed ***
exit /b 1
