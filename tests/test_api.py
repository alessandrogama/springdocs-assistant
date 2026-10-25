"""Unit tests for FastAPI endpoints (/health, /ingest, /ask, /) and lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.main as api_main
from src.api.main import app, lifespan
from src.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    IngestResponse,
)
from src.ingestion.models import IngestionStats
from src.rag.models import RetrievedChunk


@pytest.mark.anyio
async def test_root_endpoint() -> None:
    """Validate GET / returns service information."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SpringDocs Assistant"
        assert "/docs" in data["docs_url"]


@pytest.mark.anyio
async def test_lifespan_initialization_and_shutdown() -> None:
    """Validate lifespan initializes and closes singletons cleanly."""
    async with lifespan(app):
        assert api_main.hybrid_retriever is not None
        assert api_main.ollama_generator is not None
        assert api_main.ingestion_pipeline is not None

    assert api_main.hybrid_retriever is None
    assert api_main.ollama_generator is None
    assert api_main.ingestion_pipeline is None


@pytest.mark.anyio
async def test_health_endpoint_success() -> None:
    """Validate /health returns HTTP 200 with 'ok' and connected services."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_resp

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__.return_value = mock_client_instance
    mock_client_ctx.__aexit__.return_value = None

    with patch("src.api.main.httpx.AsyncClient", return_value=mock_client_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            health = HealthResponse(**data)
            assert health.status == "ok"
            assert health.qdrant == "connected"
            assert health.ollama == "connected"


@pytest.mark.anyio
async def test_health_endpoint_service_degraded() -> None:
    """Validate /health returns HTTP 503 when an external service is unavailable."""
    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = Exception("Connection refused")

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__.return_value = mock_client_instance
    mock_client_ctx.__aexit__.return_value = None

    with patch("src.api.main.httpx.AsyncClient", return_value=mock_client_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"
            assert data["qdrant"] == "disconnected"
            assert data["ollama"] == "disconnected"
            assert "details" in data


@pytest.mark.anyio
async def test_health_endpoint_returns_503_on_http_error_codes() -> None:
    """Validate /health returns 503 when downstream returns HTTP 500."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_resp

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__.return_value = mock_client_instance
    mock_client_ctx.__aexit__.return_value = None

    with patch("src.api.main.httpx.AsyncClient", return_value=mock_client_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"


@pytest.mark.anyio
async def test_ingest_endpoint_success() -> None:
    """Validate /ingest executes pipeline and returns IngestResponse."""
    mock_stats = IngestionStats(
        total_documents=9,
        total_chunks=23,
        chunks_with_code=11,
        inserted=23,
        updated=0,
        skipped=0,
        duration_seconds=0.85,
    )

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_stats

    with patch.object(api_main, "ingestion_pipeline", mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ingest", json={"force": True})
            assert response.status_code == 200
            data = response.json()
            ingest_res = IngestResponse(**data)
            assert ingest_res.inserted == 23
            assert ingest_res.updated == 0
            assert ingest_res.skipped == 0
            assert ingest_res.total_chunks == 23


@pytest.mark.anyio
async def test_ingest_endpoint_failure_raises_500() -> None:
    """Validate /ingest returns HTTP 500 when pipeline raises an exception."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = RuntimeError("Database down")

    with patch.object(api_main, "ingestion_pipeline", mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ingest")
            assert response.status_code == 500
            assert "Database down" in response.text


@pytest.mark.anyio
async def test_ask_endpoint_successful_generation() -> None:
    """Validate /ask returns answer with citations, chunks_used, and telemetry."""
    mock_chunks = [
        RetrievedChunk(
            id="test-chunk-1",
            content="Enabling Virtual Threads in application.yaml: spring.threads.virtual.enabled=true",
            score=0.85,
            doc_url="https://docs.spring.io/virtual-threads",
            section_title="Virtual Threads",
            module="core",
            spring_version="3.4.0",
            has_code=True,
        )
    ]

    mock_answer = (
        "Para habilitar virtual threads:\n"
        "```yaml\nspring.threads.virtual.enabled: true\n```\n"
        "[Fonte: Virtual Threads (https://docs.spring.io/virtual-threads)]"
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_chunks

    mock_generator = MagicMock()
    mock_generator.generate_response = AsyncMock(return_value=(mock_answer, 1250.0))

    with (
        patch.object(api_main, "hybrid_retriever", mock_retriever),
        patch.object(api_main, "ollama_generator", mock_generator),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ask",
                json={"question": "Como habilitar virtual threads no Spring Boot 3.4?"},
            )
            assert response.status_code == 200
            data = response.json()
            ask_res = AskResponse(**data)
            assert ask_res.abstained is False
            assert "virtual threads" in ask_res.answer.lower()
            assert len(ask_res.citations) == 1
            assert isinstance(ask_res.citations[0], Citation)
            assert ask_res.citations[0].title == "Virtual Threads"
            assert ask_res.citations[0].url == "https://docs.spring.io/virtual-threads"
            assert len(ask_res.chunks_used) == 1
            assert ask_res.telemetry.generation_ms == 1250.0
            assert ask_res.telemetry.model == "qwen2.5:3b"


@pytest.mark.anyio
async def test_ask_endpoint_empty_chunks_abstention() -> None:
    """Validate /ask short-circuits with abstained=True when no chunks are found."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch.object(api_main, "hybrid_retriever", mock_retriever):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ask",
                json={"question": "Como configurar Kubernetes na AWS com Terraform?"},
            )
            assert response.status_code == 200
            data = response.json()
            ask_res = AskResponse(**data)
            assert ask_res.abstained is True
            assert "não encontrei" in ask_res.answer.lower()
            assert ask_res.citations == []
            assert ask_res.chunks_used == []
            assert ask_res.telemetry.generation_ms == 0.0


@pytest.mark.anyio
async def test_ask_endpoint_empty_question_raises_422() -> None:
    """Validate /ask returns HTTP 422 when question is whitespace or empty."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "   "})
        assert response.status_code == 422


@pytest.mark.anyio
async def test_ask_endpoint_generator_failure_raises_500() -> None:
    """Validate /ask returns HTTP 500 when generator raises an exception."""
    mock_chunk = RetrievedChunk(
        id="c-1",
        content="Some content",
        score=0.9,
        doc_url="https://doc",
        section_title="Sec",
        module="core",
        spring_version="3.4.0",
        has_code=False,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_chunk]

    mock_generator = MagicMock()
    mock_generator.generate_response = AsyncMock(side_effect=RuntimeError("Ollama crashed"))

    with (
        patch.object(api_main, "hybrid_retriever", mock_retriever),
        patch.object(api_main, "ollama_generator", mock_generator),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ask", json={"question": "Test question"})
            assert response.status_code == 500
            assert "Ollama generation failure" in response.text


def test_ask_request_query_reconciliation() -> None:
    """Validate that AskRequest accepts both 'question' and 'query'."""
    req1 = AskRequest(question="Como configurar rest client?")
    assert req1.question == "Como configurar rest client?"
    assert req1.query == "Como configurar rest client?"

    req2 = AskRequest(query="Como configurar actuator?")
    assert req2.question == "Como configurar actuator?"
    assert req2.query == "Como configurar actuator?"
