"""FastAPI application entrypoint for SpringDocs Assistant."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.rag.models import AskRequest, AskResponse
from src.rag.service import RagService


class ServiceStatus(BaseModel):
    """Status details for an external dependent service."""

    status: str = Field(description="Operational status of the dependency (ok, unreachable, error)")
    details: str | None = Field(
        default=None,
        description="Additional status message or error details",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(description="Overall API status (ok or degraded)")
    version: str = Field(default="0.1.0", description="Application version")
    qdrant: ServiceStatus = Field(description="Qdrant connection health")
    ollama: ServiceStatus = Field(description="Ollama connection health")


rag_service: RagService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown routines."""
    global rag_service
    rag_service = RagService()
    yield
    rag_service = None


app: FastAPI = FastAPI(
    title="SpringDocs Assistant API",
    description=(
        "Production-grade RAG and fine-tuning assistant for official Spring Boot 3.x documentation"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler returning structured error JSON."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint welcoming requests."""
    return {
        "service": "SpringDocs Assistant",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Monitoring"],
)
async def health() -> HealthResponse:
    """Verify connectivity with Qdrant vector database and Ollama service."""
    # Check Qdrant
    qdrant_status: ServiceStatus
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            headers: dict[str, str] = {}
            if settings.qdrant_api_key:
                headers["api-key"] = settings.qdrant_api_key
            resp = await client.get(f"{settings.qdrant_url}/collections", headers=headers)
            if resp.status_code == 200:
                qdrant_status = ServiceStatus(status="ok")
            else:
                qdrant_status = ServiceStatus(
                    status="error",
                    details=f"Returned HTTP {resp.status_code}",
                )
    except Exception as exc:  # noqa: BLE001
        qdrant_status = ServiceStatus(status="unreachable", details=str(exc))

    # Check Ollama
    ollama_status: ServiceStatus
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            if resp.status_code == 200:
                ollama_status = ServiceStatus(status="ok")
            else:
                ollama_status = ServiceStatus(
                    status="error",
                    details=f"Returned HTTP {resp.status_code}",
                )
    except Exception as exc:  # noqa: BLE001
        ollama_status = ServiceStatus(status="unreachable", details=str(exc))

    is_healthy = qdrant_status.status == "ok" and ollama_status.status == "ok"
    overall_status = "ok" if is_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        qdrant=qdrant_status,
        ollama=ollama_status,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
)
async def ask(request: AskRequest) -> AskResponse:
    """Answer technical Spring Boot question using hybrid RAG retrieval."""
    global rag_service
    if rag_service is None:
        rag_service = RagService()

    try:
        return await rag_service.answer_question(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {exc}",
        ) from exc
