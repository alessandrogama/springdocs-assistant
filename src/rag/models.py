"""Pydantic schemas for RAG retrieval and question answering."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Input payload for technical query answering."""

    query: str = Field(
        ...,
        min_length=3,
        description="User question about Spring Boot documentation",
        examples=["Como configurar pool de conexões HikariCP no Spring Boot 3?"],
    )
    spring_version: str | None = Field(
        default=None,
        description="Target Spring Boot version filter (e.g. '3.4'). Defaults to configured version.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature",
    )
    model_name: str | None = Field(
        default=None,
        description="Ollama model override",
    )
    rerank_enabled: bool | None = Field(
        default=None,
        description="Enable cross-encoder reranking flag",
    )


class RetrievedChunk(BaseModel):
    """Metadata and content of a retrieved context chunk."""

    id: str = Field(description="Deterministic chunk ID")
    content: str = Field(description="Chunk text and code snippet")
    score: float = Field(description="Retrieval similarity or RRF score")
    doc_url: str = Field(description="Canonical documentation URL")
    section_title: str = Field(description="Section heading")
    module: str = Field(description="Spring Boot module")
    spring_version: str = Field(description="Spring Boot version")
    has_code: bool = Field(description="Indicates if chunk contains code")


class AskResponse(BaseModel):
    """Structured response for RAG query."""

    answer: str = Field(description="Generated technical response in pt-BR")
    citations: list[str] = Field(
        default_factory=list,
        description="List of cited documentation sections and URLs in format '[Fonte: Título (URL)]'",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks retrieved from vector store",
    )
    retrieval_latency_ms: float = Field(
        description="Time taken to search and retrieve chunks in ms"
    )
    generation_latency_ms: float = Field(description="Time taken for LLM inference in ms")
    total_latency_ms: float = Field(description="Total end-to-end processing time in ms")
    model_name: str = Field(description="Model used for generation")
    spring_version: str = Field(description="Target Spring Boot version used for filtering")
    is_abstained: bool = Field(
        default=False,
        description="Indicates whether the assistant abstained due to insufficient documentation context",
    )
