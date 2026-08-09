@echo off
chcp 65001 >nul
cd /d %~dp0

echo [1/3] 创建虚拟环境 .venv ...
py -3 -m venv .venv
call .venv\Scripts\activate.bat

echo [2/3] 安装 PyTorch（CPU 版，约 200MB，视网速需要几分钟）...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [3/3] 安装其他依赖 ...
pip install -r requirements.txt

echo.
echo ============================================
echo 安装完成！接下来：
echo   1) 双击 train.bat 训练模型
echo   2) 双击 run_demo.bat 打开手写识别演示
echo ============================================
pause
