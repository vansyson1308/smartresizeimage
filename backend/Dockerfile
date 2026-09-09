FROM python:3.11-slim

WORKDIR /app

# System dependencies for Pillow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0t64 \
    tesseract-ocr \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py ./backend/__init__.py
COPY app/ ./backend/app/
RUN mkdir -p /app/data

ENV AUTOBANNER_DATA_DIR=/app/data \
    OMP_THREAD_LIMIT=1

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default: the project API + web UI. The legacy Gradio interface remains available with
#   python -m backend.app.main
CMD ["uvicorn", "backend.app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
