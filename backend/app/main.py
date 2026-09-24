"""Server entry point: ``python -m app.main`` serves the web studio and REST API.

The studio is available at ``http://localhost:7860`` and the interactive API
reference at ``/docs``.
"""

from __future__ import annotations

import logging
import os
import sys

from .logging_config import setup_logging

logger = logging.getLogger("autobanner.main")


def _env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def main(host: str | None = None, port: int | None = None) -> None:
    """Start the HTTP server (studio + API)."""
    setup_logging(
        os.environ.get("AUTOBANNER_LOG_LEVEL", "INFO"),
        json_logs=os.environ.get("AUTOBANNER_LOG_FORMAT", "text").lower() == "json",
    )

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is required to serve the app: pip install -r requirements.txt")
        sys.exit(1)

    from .api import create_app

    host = host or _env("AUTOBANNER_SERVER_NAME", "GRADIO_SERVER_NAME", default="127.0.0.1")
    port = port or int(_env("AUTOBANNER_SERVER_PORT", "GRADIO_SERVER_PORT", "PORT",
                            default="7860"))

    app = create_app()
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    logger.info("AutoBanner studio: http://%s:%d  (API docs: /docs)", display_host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        proxy_headers=app.state.settings.trust_proxy_headers,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
