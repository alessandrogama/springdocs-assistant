"""Unit tests for RagService, HybridRetriever and re-exports."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import src.main as app_main
from src.rag.generator import OllamaGenerator
from src.rag.models import AskRequest, RetrievedChunk
from src.rag.retriever import HybridRetriever
from src.rag.service import RagService


def test_main_reexports() -> None:
    """Validate src.main re-exports expected FastAPI components."""
    assert hasattr(app_main, "app")
    assert hasattr(app_main, "health")
    assert hasattr(app_main, "ask")
    assert hasattr(app_main, "ingest")


@pytest.mark.anyio
async def test_rag_service_empty_chunks_abstention() -> None:
    """Validate RagService immediately abstains when retriever returns zero chunks."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_generator = MagicMock()

    service = RagService(retriever=mock_retriever, generator=mock_generator)
    req = AskRequest(query="Pergunta fora de contexto")

    response = await service.answer_question(req)
    assert response.is_abstained is True
    assert "não encontrei" in response.answer.lower()
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert response.generation_latency_ms == 0.0
    mock_generator.generate_response.assert_not_called()


@pytest.mark.anyio
async def test_rag_service_successful_grounded_answer() -> None:
    """Validate RagService generates grounded answer and extracts citations."""
    mock_chunk = RetrievedChunk(
        id="c-1",
        content="Virtual threads improve throughput.",
        score=0.9,
        doc_url="https://docs.spring.io/threads",
        section_title="Virtual Threads",
        module="core",
        spring_version="3.4.0",
        has_code=False,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [mock_chunk]

    mock_generator = MagicMock()
    mock_generator.generate_response = AsyncMock(
        return_value=(
            "Resposta sobre threads.\n[Fonte: Virtual Threads (https://docs.spring.io/threads)]",
            450.0,
        )
    )

    service = RagService(retriever=mock_retriever, generator=mock_generator)
    req = AskRequest(query="Como usar virtual threads?")

    response = await service.answer_question(req)
    assert response.is_abstained is False
    assert len(response.citations) == 1
    assert "Virtual Threads" in response.citations[0]
    assert response.generation_latency_ms == 450.0
    assert len(response.retrieved_chunks) == 1


def test_hybrid_retriever_query_points() -> None:
    """Validate HybridRetriever builds prefetch queries and handles response."""
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.id = "p-1"
    mock_point.score = 0.88
    mock_point.payload = {
        "content": "Test chunk content",
        "doc_url": "https://test.com",
        "section_title": "Section A",
        "module": "web",
        "spring_version": "3.4.0",
        "has_code": True,
    }

    mock_query_result = MagicMock()
    mock_query_result.points = [mock_point]
    mock_client.query_points.return_value = mock_query_result

    mock_embedder = MagicMock()
    mock_sparse = MagicMock()
    mock_sparse.indices = [1]
    mock_sparse.values = [0.5]
    mock_embedder.embed_text.return_value = ([0.1] * 384, mock_sparse)

    retriever = HybridRetriever(
        client=mock_client,
        embedder=mock_embedder,
        collection_name="test_col",
    )

    results = retriever.retrieve("test query", spring_version="3.4.0", top_k=2)
    assert len(results) == 1
    assert results[0].id == "p-1"
    assert results[0].section_title == "Section A"
    assert results[0].has_code is True


@pytest.mark.anyio
async def test_ollama_generator_client_close() -> None:
    """Validate OllamaGenerator.close runs cleanly."""
    generator = OllamaGenerator()
    await generator.close()

