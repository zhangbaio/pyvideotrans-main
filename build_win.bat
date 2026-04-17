@echo off
REM ============================================================
REM pyVideoTrans Windows 打包脚本
REM 前置:
REM   1. 安装 Python 3.10 (必须 <3.11, 项目约束)
REM   2. 安装 uv:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
REM 运行:
REM   双击本文件, 或在项目根目录 cmd 执行 build_win.bat
REM ============================================================

setlocal
cd /d "%~dp0"

echo [1/5] 同步依赖 (uv sync)...
uv sync
if errorlevel 1 goto :err

echo [2/5] 清理旧产物...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo [3/5] PyInstaller 打包...
uv run pyinstaller pyvideotrans.spec --clean --noconfirm --contents-directory=.
if errorlevel 1 goto :err

echo [4/5] 创建空运行时目录 + 模型下载说明...
set OUT=dist\pyVideoTrans
mkdir "%OUT%\models"       2>nul
mkdir "%OUT%\output"       2>nul
mkdir "%OUT%\logs"         2>nul
mkdir "%OUT%\tmp"          2>nul
mkdir "%OUT%\presets_user" 2>nul
copy /y "models\下载说明.txt" "%OUT%\models\下载说明.txt" >nul 2>&1

echo [5/5] 完成!
echo.
echo 产物目录: %CD%\%OUT%
echo 入口: %CD%\%OUT%\pyVideoTrans.exe
echo.
echo 首次运行前, 用户需按 models\下载说明.txt 下载模型到 models\ 目录
pause
exit /b 0

:err
echo.
echo *** 打包失败 ***
pause
exit /b 1
