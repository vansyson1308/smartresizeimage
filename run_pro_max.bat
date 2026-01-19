@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo BANNER RESIZER PRO MAX - LAUNCHER
echo ================================================
echo.

REM Env vars
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
set "TF_CPP_MIN_LOG_LEVEL=3"

REM Cache folders (optional)
set "HF_HOME=%cd%\cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"

echo Creating directories...
if not exist "models" mkdir "models"
if not exist "cache" mkdir "cache"
if not exist "outputs" mkdir "outputs"

echo.

if not exist "venv_ai\Scripts\python.exe" (
  echo [ERROR] venv_ai not found. Please run install_pro_max.bat first.
  pause
  exit /b 1
)

call "venv_ai\Scripts\activate"

echo Launching app...
echo Open: http://127.0.0.1:7860
echo.

python app_pro_max.py

pause
