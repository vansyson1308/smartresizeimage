# syntax=docker/dockerfile:1
# AutoBanner - studio + REST API.
#   docker build -t autobanner .                          # lite (default, ~500 MB)
#   docker build -t autobanner:ai --build-arg WITH_AI=true .  # + CLIP/LaMa (several GB)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG WITH_AI=false
COPY requirements.txt requirements-ai.txt ./
RUN if [ "$WITH_AI" = "true" ]; then pip install -r requirements-ai.txt; \
    else pip install -r requirements.txt; fi

RUN useradd --create-home --uid 10001 autobanner \
    && mkdir -p /app/cache /data \
    && chown autobanner:autobanner /app/cache /data

COPY --chown=autobanner:autobanner app/ ./app/

USER autobanner

# Port: 7860 by default, or $PORT when the platform injects one.
ENV AUTOBANNER_SERVER_NAME=0.0.0.0 \
    AUTOBANNER_USE_AI=${WITH_AI} \
    AUTOBANNER_DATA_DIR=/data \
    HF_HOME=/app/cache

# Job results; a named/anonymous volume keeps it writable under --read-only.
VOLUME ["/data"]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request; port=os.environ.get('AUTOBANNER_SERVER_PORT') or os.environ.get('PORT') or '7860'; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "app.main"]
