"""Generation package for SpringDocs Assistant."""

from src.generation.generator import OllamaGenerator
from src.generation.models import (
    GenerationResult,
    OllamaClientError,
    OllamaConnectionError,
    OllamaTimeoutError,
)
from src.generation.prompts import (
    STANDARD_ABSTENTION_PHRASE,
    build_prompts,
    extract_and_validate_citations,
    is_abstention_text,
)

__all__ = [
    "OllamaGenerator",
    "GenerationResult",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "STANDARD_ABSTENTION_PHRASE",
    "build_prompts",
    "extract_and_validate_citations",
    "is_abstention_text",
]
