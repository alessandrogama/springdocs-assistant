"""Data models and domain exceptions for the generation module."""

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """Structured response returned by the Ollama generation engine."""

    answer: str = Field(description="Generated technical text in pt-BR")
    citations: list[str] = Field(
        default_factory=list,
        description="List of verified citations in format '[Fonte: Título (URL)]'",
    )
    abstained: bool = Field(
        default=False,
        description="Whether the generator safely abstained due to lack of context",
    )
    model_used: str = Field(description="Name or tag of the Ollama model used")
    generation_latency_ms: float = Field(description="Inference time in milliseconds")


class OllamaClientError(Exception):
    """Base exception for errors communicating with the Ollama service."""


class OllamaConnectionError(OllamaClientError):
    """Exception raised when Ollama service is unreachable or connection fails."""


class OllamaTimeoutError(OllamaClientError):
    """Exception raised when an Ollama request times out."""
