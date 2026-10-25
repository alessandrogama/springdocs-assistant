"""Data models for vector retrieval."""

from typing import NamedTuple

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Structured chunk returned by the retrieval engine."""

    chunk_id: str = Field(description="Deterministic chunk UUID")
    text: str = Field(description="Chunk textual content")
    doc_url: str = Field(description="Canonical documentation URL")
    section_title: str = Field(description="Section heading or title")
    spring_version: str = Field(description="Spring Boot version")
    score: float = Field(description="Retrieval similarity score")
    module: str = Field(default="core", description="Spring Boot module")
    has_code: bool = Field(default=False, description="Flag indicating if code is present")

    @property
    def id(self) -> str:
        """Alias for chunk_id."""
        return self.chunk_id

    @property
    def content(self) -> str:
        """Alias for text."""
        return self.text


class RetrievalResult(NamedTuple):
    """Retrieval operation result containing chunks and measured latency in ms."""

    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
