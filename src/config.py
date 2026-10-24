"""Application configuration module using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for SpringDocs Assistant.

    Loads defaults or reads overrides from environment variables and an optional .env file.
    All properties are strictly typed with production-grade defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Service URLs
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional API key for Qdrant instance",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama local LLM API URL",
    )

    # Collection and Spring Boot metadata
    collection_name: str = Field(
        default="spring_docs_v3_4",
        description="Name of the Qdrant collection for indexed chunks",
    )
    spring_version: str = Field(
        default="3.4",
        description="Spring Boot documentation version to index and query",
    )

    # Embedding model
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "Embedding model name "
            "(e.g. BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small, BAAI/bge-m3)"
        ),
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimension of dense vectors (384 for bge-small/e5-small, 1024 for bge-m3)",
    )

    # LLM generation configuration
    llm_model: str = Field(
        default="qwen2.5:3b",
        description="Ollama model tag to use for answer generation",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Request timeout in seconds for Ollama LLM calls",
    )

    # Retrieval hyperparameters
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks retrieved per query",
    )
    score_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for retrieved chunks",
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Flag to enable cross-encoder reranking over retrieved chunks",
    )

    # API server settings
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host binding",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI port binding",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag",
    )


settings: Settings = Settings()
