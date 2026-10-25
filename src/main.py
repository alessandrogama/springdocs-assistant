"""FastAPI application entrypoint for SpringDocs Assistant (re-exports src.api.main)."""

from src.api.main import (
    app,
    ask,
    global_exception_handler,
    health,
    ingest,
    lifespan,
    root,
)
from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "app",
    "ask",
    "global_exception_handler",
    "health",
    "ingest",
    "lifespan",
    "root",
]
