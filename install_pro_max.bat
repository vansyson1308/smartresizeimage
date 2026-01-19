@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo BANNER RESIZER PRO MAX - INSTALL
echo ================================================
echo.

set "VENV=venv_ai"

echo [1/5] Creating venv if needed...
if not exist "%VENV%\Scripts\python.exe" (
  python -m venv "%VENV%"
)

echo [2/5] Activating venv...
call "%VENV%\Scripts\activate"

echo [3/5] Upgrading pip tools...
python -m pip install --upgrade pip setuptools wheel

echo [4/5] Installing torch + xformers...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install xformers --index-url https://download.pytorch.org/whl/cu118

echo [5/5] Installing libraries...
python -m pip install opencv-python==4.8.1.78 Pillow==10.1.0 numpy==1.24.3 scipy==1.11.4 scikit-image==0.22.0 scikit-learn==1.3.2
python -m pip install diffusers==0.24.0 transformers==4.35.2 accelerate==0.25.0 safetensors==0.4.1
python -m pip install "huggingface_hub==0.19.4"
python -m pip install invisible-watermark supervision==0.15.0 segment-anything-py==1.0 gradio==4.13.0

echo.
echo Verify versions:
python -c "import gradio, huggingface_hub; print('gradio:', gradio.__version__); print('huggingface_hub:', huggingface_hub.__version__)"

echo.
echo DONE.
pause
