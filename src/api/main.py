"""FastAPI application entrypoint for SpringDocs Assistant."""

import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkPreview,
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Telemetry,
)
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.rag.generator import OllamaGenerator
from src.rag.models import RetrievedChunk
from src.rag.prompt import (
    ABSTENTION_PHRASE,
    build_rag_prompt,
    extract_citations,
    is_abstention_response,
)
from src.rag.retriever import HybridRetriever

# Module-level singletons managed by lifespan
hybrid_retriever: HybridRetriever | None = None
ollama_generator: OllamaGenerator | None = None
ingestion_pipeline: IngestionPipeline | None = None


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Parse raw citation strings into structured Citation objects."""
    raw_citations = extract_citations(text, chunks)
    pattern = re.compile(r"\[Fonte:\s*([^\]]+?)\s*\((https?://[^\)]+)\)\]")
    parsed: list[Citation] = []
    seen: set[str] = set()

    for raw in raw_citations:
        match = pattern.search(raw)
        if match:
            title = match.group(1).strip()
            url = match.group(2).strip()
            if url not in seen:
                seen.add(url)
                parsed.append(Citation(title=title, url=url, raw=raw))
        else:
            if raw not in seen:
                seen.add(raw)
                parsed.append(Citation(title=raw, url="", raw=raw))

    return parsed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager initializing dependencies."""
    global hybrid_retriever, ollama_generator, ingestion_pipeline
    hybrid_retriever = HybridRetriever()
    ollama_generator = OllamaGenerator()
    ingestion_pipeline = IngestionPipeline()
    yield
    if ollama_generator:
        await ollama_generator.close()
    hybrid_retriever = None
    ollama_generator = None
    ingestion_pipeline = None


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
    tags=["Monitoring"],
)
async def health(response: Response) -> HealthResponse | JSONResponse:
    """Verify real-time connectivity with Qdrant vector database and Ollama service."""
    details: dict[str, str] = {}
    qdrant_status = "disconnected"
    ollama_status = "disconnected"

    # 1. Check Qdrant
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            headers: dict[str, str] = {}
            if settings.qdrant_api_key:
                headers["api-key"] = settings.qdrant_api_key
            q_resp = await client.get(f"{settings.qdrant_url}/collections", headers=headers)
            if q_resp.status_code == 200:
                qdrant_status = "connected"
            else:
                details["qdrant"] = f"HTTP {q_resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        details["qdrant"] = f"Connection error: {exc}"

    # 2. Check Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            o_resp = await client.get(f"{settings.ollama_url}/api/tags")
            if o_resp.status_code == 200:
                ollama_status = "connected"
            else:
                details["ollama"] = f"HTTP {o_resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        details["ollama"] = f"Connection error: {exc}"

    is_healthy = qdrant_status == "connected" and ollama_status == "connected"

    if is_healthy:
        response.status_code = status.HTTP_200_OK
        return HealthResponse(
            status="ok",
            qdrant=qdrant_status,
            ollama=ollama_status,
            details=None,
        )

    # Return HTTP 503 if any dependent service is down
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "degraded",
            "qdrant": qdrant_status,
            "ollama": ollama_status,
            "details": details,
        },
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["Ingestion"],
)
async def ingest(request: IngestRequest | None = None) -> IngestResponse:
    """Trigger idempotent ingestion and index update in Qdrant."""
    global ingestion_pipeline
    if ingestion_pipeline is None:
        ingestion_pipeline = IngestionPipeline()

    force_fetch = request.force if request else False
    try:
        stats = ingestion_pipeline.run(force_fetch=force_fetch)
        return IngestResponse(
            inserted=stats.inserted,
            updated=stats.updated,
            skipped=stats.skipped,
            total_chunks=stats.total_chunks,
            duration_seconds=stats.duration_seconds,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing ingestion pipeline: {exc}",
        ) from exc


@app.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
)
async def ask(request: AskRequest) -> AskResponse:
    """Answer technical Spring Boot question using hybrid retrieval and grounding."""
    global hybrid_retriever, ollama_generator
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever()
    if ollama_generator is None:
        ollama_generator = OllamaGenerator()

    total_start = time.perf_counter()
    question = request.question or request.query or ""
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'question' cannot be empty.",
        )

    # 1. Hybrid Retrieval
    retrieval_start = time.perf_counter()
    retrieved_chunks = hybrid_retriever.retrieve(
        query=question,
        spring_version=request.spring_version or settings.spring_version,
        top_k=request.top_k or settings.top_k,
        score_threshold=request.score_threshold,
    )
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

    active_model = request.model or settings.llm_model

    # 2. Short-circuit if context is completely empty
    if not retrieved_chunks:
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        return AskResponse(
            answer=f"{ABSTENTION_PHRASE} para responder a esta pergunta.",
            abstained=True,
            citations=[],
            chunks_used=[],
            telemetry=Telemetry(
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
                model=active_model,
            ),
        )

    # 3. Format context chunks
    chunks_used = [
        ChunkPreview(
            chunk_id=c.id,
            section_title=c.section_title,
            doc_url=c.doc_url,
            score=c.score,
            module=c.module,
            has_code=c.has_code,
            text_preview=c.content[:200] + ("..." if len(c.content) > 200 else ""),
        )
        for c in retrieved_chunks
    ]

    # 4. Strict grounding prompt
    system_prompt, user_prompt = build_rag_prompt(
        query=question,
        chunks=retrieved_chunks,
    )

    # 5. LLM Inference
    try:
        answer_text, generation_ms = await ollama_generator.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=request.model,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ollama generation failure: {exc}",
        ) from exc

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    abstained = is_abstention_response(answer_text)
    citations = [] if abstained else parse_citations(answer_text, retrieved_chunks)

    return AskResponse(
        answer=answer_text,
        abstained=abstained,
        citations=citations,
        chunks_used=chunks_used,
        telemetry=Telemetry(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            model=active_model,
        ),
    )
