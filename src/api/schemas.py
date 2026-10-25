"""Pydantic schemas for the SpringDocs Assistant FastAPI application."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """Citation reference to official Spring Boot documentation."""

    title: str = Field(description="Section or documentation title")
    url: str = Field(description="Canonical documentation URL")
    raw: str = Field(default="", description="Full citation string, e.g. [Fonte: Título (URL)]")


class ChunkPreview(BaseModel):
    """Context chunk used by the RAG model."""

    chunk_id: str = Field(description="Unique deterministic chunk ID")
    section_title: str = Field(description="Title of documentation section")
    doc_url: str = Field(description="URL of source document")
    score: float = Field(description="Similarity or RRF retrieval score")
    module: str = Field(default="core", description="Spring Boot module")
    has_code: bool = Field(default=False, description="Flag indicating if chunk contains code")
    text_preview: str = Field(default="", description="Short snippet of the chunk content")


class Telemetry(BaseModel):
    """Execution latency and model telemetry."""

    retrieval_ms: float = Field(
        description="Time taken for vector and hybrid retrieval in milliseconds"
    )
    generation_ms: float = Field(description="Time taken for LLM inference in milliseconds")
    total_ms: float = Field(description="Total end-to-end request processing time in milliseconds")
    model: str = Field(description="LLM model identifier used for generation")


class AskRequest(BaseModel):
    """Payload for technical Spring Boot queries."""

    question: str = Field(
        default="",
        description="User technical question about Spring Boot documentation",
        examples=["Como configurar chamadas síncronas com RestClient no Spring Boot 3.4?"],
    )
    query: str | None = Field(
        default=None,
        description="Alias for question to ensure full backward compatibility",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context",
    )
    spring_version: str | None = Field(
        default="3.4.0",
        description="Target Spring Boot version filter (defaults to 3.4.0)",
    )
    model: str | None = Field(
        default=None,
        description="Ollama model override (e.g. qwen2.5:3b)",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score threshold",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature override",
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_question_and_query(cls, data: Any) -> Any:
        """Allow either 'question' or 'query' to be used interchangeably."""
        if isinstance(data, dict):
            if not data.get("question") and data.get("query"):
                data["question"] = data["query"]
            elif not data.get("query") and data.get("question"):
                data["query"] = data["question"]
        return data


class AskResponse(BaseModel):
    """Structured response for RAG query."""

    answer: str = Field(description="Generated technical response in pt-BR")
    abstained: bool = Field(
        description="True if model abstained due to insufficient context or out-of-scope query"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of cited documentation sections and URLs",
    )
    chunks_used: list[ChunkPreview] = Field(
        default_factory=list,
        description="List of chunks provided in LLM context",
    )
    telemetry: Telemetry = Field(description="Latency measurements and model metadata")


class IngestRequest(BaseModel):
    """Optional parameters for ingestion pipeline execution."""

    force: bool = Field(
        default=False,
        description="If True, re-fetches and re-chunks documents before idempotent upsert",
    )


class IngestResponse(BaseModel):
    """Result summary of idempotent ingestion execution."""

    inserted: int = Field(description="Number of newly inserted chunks")
    updated: int = Field(description="Number of updated chunks")
    skipped: int = Field(description="Number of skipped/unchanged chunks")
    total_chunks: int = Field(default=0, description="Total chunks evaluated")
    duration_seconds: float = Field(default=0.0, description="Execution duration in seconds")


class HealthResponse(BaseModel):
    """Health check response for monitoring."""

    status: str = Field(description="Overall status ('ok' or 'degraded')")
    qdrant: str = Field(description="Qdrant status ('connected' or 'disconnected')")
    ollama: str = Field(description="Ollama status ('connected' or 'disconnected')")
    details: dict[str, str] | None = Field(
        default=None,
        description="Optional diagnostic error details when unhealthy",
    )
