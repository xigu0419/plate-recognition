@echo off
REM ============================
REM  车牌识别环境一键配置脚本
REM  请在 Anaconda Prompt 中运行
REM ============================
chcp 65001 >nul
echo.
echo ========================================
echo   车牌识别环境一键配置
echo ========================================
echo.

REM 创建并激活环境
echo [1/3] 创建 conda 环境 "plate" ...
call conda create -n plate python=3.9 -y

echo.
echo [2/3] 激活环境 ...
call conda activate plate

echo.
echo [3/3] 安装依赖库 ...
call pip install -r requirements.txt
call pip install --upgrade pip

echo.
echo ========================================
echo   环境配置完成！
echo.
echo   使用步骤：
echo   1. 每次运行前先激活环境：conda activate plate
echo   2. 安装 Tesseract OCR（重要！）
echo      下载地址：https://github.com/UB-Mannheim/tesseract/wiki
echo      安装后把 tesseract.exe 的路径填到 plate_recognition.py 的
echo      TESSERACT_CMD 变量中
echo   3. 把车牌照片放到 sample_images\ 文件夹
echo   4. 运行：python plate_recognition.py
echo ========================================
echo.
pause
