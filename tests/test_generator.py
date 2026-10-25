"""Unit tests for OllamaGenerator with mocked HTTP responses."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.generation.generator import OllamaGenerator
from src.generation.models import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaTimeoutError,
)
from src.generation.prompts import STANDARD_ABSTENTION_PHRASE
from src.retrieval.models import RetrievedChunk


def _create_sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-001",
        text="HikariCP is the default connection pool in Spring Boot 3.x.",
        doc_url="https://docs.spring.io/spring-boot/reference/data/sql.html",
        section_title="Connection Pool Auto-configuration",
        spring_version="3.4",
        score=0.95,
        module="data",
        has_code=False,
    )


def test_generator_short_circuits_on_empty_chunks() -> None:
    """Ensure generator immediately abstains when 0 context chunks are provided under RAG mode."""
    generator = OllamaGenerator()

    result = asyncio.run(
        generator.generate(
            query="Como configurar o banco?",
            chunks=[],
            use_rag=True,
        )
    )

    assert result.abstained is True
    assert result.answer == STANDARD_ABSTENTION_PHRASE
    assert result.citations == []
    assert result.generation_latency_ms == 0.0


def test_generator_extracts_valid_citations_and_rejects_hallucinations() -> None:
    """Verify generator keeps valid chunk citations and discards hallucinated/invented URLs."""
    chunk = _create_sample_chunk()
    generator = OllamaGenerator()

    # Model response with one valid URL (from chunk) and one invented hallucinated URL
    llm_output = (
        "No Spring Boot 3, o pool padrão é o HikariCP "
        "[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)].\n"
        "Mais detalhes podem ser vistos em "
        "[Fonte: Fake Hallucination (https://hallucinated-invented-docs.com/fake)]."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"message": {"content": llm_output}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        result = asyncio.run(
            generator.generate(
                query="Qual o pool padrão?",
                chunks=[chunk],
                use_rag=True,
            )
        )

        assert result.abstained is False
        assert len(result.citations) == 1
        # The valid citation matching context chunk must be present
        assert (
            result.citations[0]
            == "[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)]"
        )
        # The hallucinated URL must NEVER be admitted
        assert "https://hallucinated-invented-docs.com/fake" not in str(result.citations)


def test_generator_abstention_when_model_refuses() -> None:
    """Verify that when the LLM returns the standard refusal, abstained is True and citations are empty."""
    chunk = _create_sample_chunk()
    generator = OllamaGenerator()

    mock_resp = httpx.Response(
        status_code=200,
        json={"message": {"content": STANDARD_ABSTENTION_PHRASE}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        result = asyncio.run(
            generator.generate(
                query="Como configurar Kubernetes Istio?",
                chunks=[chunk],
                use_rag=True,
            )
        )

        assert result.abstained is True
        assert result.citations == []
        assert STANDARD_ABSTENTION_PHRASE in result.answer


def test_generator_handles_connection_error_when_ollama_offline() -> None:
    """Verify OllamaConnectionError is raised when Ollama service is unreachable."""
    chunk = _create_sample_chunk()
    generator = OllamaGenerator()

    with patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.ConnectError("Connection refused to Ollama"),
    ):
        with pytest.raises(OllamaConnectionError) as exc_info:
            asyncio.run(
                generator.generate(
                    query="Pergunta teste",
                    chunks=[chunk],
                    use_rag=True,
                )
            )

        assert "Failed to connect to Ollama" in str(exc_info.value)


def test_generator_handles_timeout() -> None:
    """Verify OllamaTimeoutError is raised when inference exceeds timeout."""
    chunk = _create_sample_chunk()
    generator = OllamaGenerator(timeout_seconds=5.0)

    with patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.TimeoutException("Read timeout"),
    ):
        with pytest.raises(OllamaTimeoutError) as exc_info:
            asyncio.run(
                generator.generate(
                    query="Pergunta teste",
                    chunks=[chunk],
                    use_rag=True,
                )
            )

        assert "Timeout of" in str(exc_info.value)


def test_generator_handles_http_status_error() -> None:
    """Verify OllamaClientError is raised on 500 error from Ollama."""
    chunk = _create_sample_chunk()
    generator = OllamaGenerator()

    mock_response = httpx.Response(status_code=500, text="Internal Server Error")
    mock_error = httpx.HTTPStatusError(
        message="500 Internal Server Error",
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        response=mock_response,
    )

    with patch("httpx.AsyncClient.post", side_effect=mock_error):
        with pytest.raises(OllamaClientError) as exc_info:
            asyncio.run(
                generator.generate(
                    query="Pergunta teste",
                    chunks=[chunk],
                    use_rag=True,
                )
            )

        assert "Ollama returned HTTP 500" in str(exc_info.value)


def test_generator_use_rag_false_calls_base_prompt_without_short_circuit() -> None:
    """Verify use_rag=False allows calling base model without context and does not short-circuit."""
    generator = OllamaGenerator()

    mock_resp = httpx.Response(
        status_code=200,
        json={"message": {"content": "Resposta do modelo base sem RAG."}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        result = asyncio.run(
            generator.generate(
                query="Como funciona a injeção de dependência no Spring?",
                chunks=[],
                use_rag=False,
            )
        )

        assert mock_post.called
        assert result.abstained is False
        assert result.answer == "Resposta do modelo base sem RAG."
        assert result.citations == []
