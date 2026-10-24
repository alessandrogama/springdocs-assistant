"""Data models for documentation ingestion and chunking."""

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Raw documentation page or document content."""

    doc_url: str = Field(description="Canonical URL of the documentation page")
    spring_version: str = Field(default="3.4", description="Spring Boot version")
    module: str = Field(description="Spring Boot module (e.g. web, data, security, core, actuator)")
    title: str = Field(description="Document title")
    content: str = Field(description="Raw HTML or Markdown content")


class DocumentChunk(BaseModel):
    """Chunk of documentation ready for vector indexing."""

    id: str = Field(description="Deterministic UUIDv5 chunk identifier")
    doc_url: str = Field(description="Canonical URL of the source section")
    spring_version: str = Field(description="Spring Boot version (e.g. 3.4)")
    section_title: str = Field(description="Title of the section or header")
    module: str = Field(description="Spring Boot module")
    content: str = Field(description="Cleaned text and code snippet content")
    content_hash: str = Field(description="SHA-256 hash of the normalized content")
    has_code: bool = Field(default=False, description="Flag indicating if chunk contains code")
    chunk_index: int = Field(default=0, ge=0, description="Sequential index within the section")
    parent_section: str | None = Field(default=None, description="Parent section title if nested")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestionStats(BaseModel):
    """Execution summary statistics for documentation ingestion."""

    total_documents: int = 0
    total_chunks: int = 0
    chunks_with_code: int = 0
    new_or_updated_chunks: int = 0
    unchanged_chunks: int = 0
    duration_seconds: float = 0.0
