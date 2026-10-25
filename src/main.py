"""FastAPI application entrypoint for SpringDocs Assistant (re-exports src.api.main)."""

from src.api.main import (
    HealthResponse,
    ServiceStatus,
    app,
    ask,
    global_exception_handler,
    health,
    lifespan,
    root,
)

__all__ = [
    "HealthResponse",
    "ServiceStatus",
    "app",
    "ask",
    "global_exception_handler",
    "health",
    "lifespan",
    "root",
]
