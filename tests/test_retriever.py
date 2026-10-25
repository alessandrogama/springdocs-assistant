"""Unit tests for DocumentRetriever filtering and score ordering."""

from unittest.mock import MagicMock

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.ingestion.embedder import SparseVectorData
from src.retrieval.retriever import DocumentRetriever


def test_build_version_filter() -> None:
    """Verify that build_version_filter formats Qdrant filter condition correctly."""
    retriever = DocumentRetriever(client=MagicMock(), embedder=MagicMock())

    # None spring_version returns None
    assert retriever.build_version_filter(None) is None

    # Specific version produces valid Filter
    filter_obj = retriever.build_version_filter("3.4")
    assert isinstance(filter_obj, Filter)
    assert len(filter_obj.must) == 1
    condition = filter_obj.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "spring_version"
    assert condition.match == MatchValue(value="3.4")


def test_retriever_score_ordering_and_filtering() -> None:
    """Verify that retrieved chunks are sorted by score descending and threshold is respected."""
    mock_client = MagicMock()
    mock_embedder = MagicMock()

    mock_embedder.embed_text.return_value = (
        [0.1] * 384,
        SparseVectorData(indices=[1, 2], values=[0.5, 0.8]),
    )

    # Simulate 3 points with scores in non-descending order
    point1 = MagicMock()
    point1.id = "uuid-1"
    point1.score = 0.35
    point1.payload = {
        "content": "Low score content",
        "doc_url": "https://docs.spring.io/1",
        "section_title": "Section 1",
        "spring_version": "3.4",
        "module": "web",
    }

    point2 = MagicMock()
    point2.id = "uuid-2"
    point2.score = 0.92
    point2.payload = {
        "content": "High score content",
        "doc_url": "https://docs.spring.io/2",
        "section_title": "Section 2",
        "spring_version": "3.4",
        "module": "security",
    }

    point3 = MagicMock()
    point3.id = "uuid-3"
    point3.score = 0.15  # below threshold 0.20
    point3.payload = {
        "content": "Too low score content",
        "doc_url": "https://docs.spring.io/3",
        "section_title": "Section 3",
        "spring_version": "3.4",
        "module": "data",
    }

    mock_response = MagicMock()
    mock_response.points = [point1, point2, point3]
    mock_client.query_points.return_value = mock_response

    retriever = DocumentRetriever(client=mock_client, embedder=mock_embedder)

    result = retriever.retrieve(
        query="Spring Security",
        top_k=5,
        score_threshold=0.20,
        spring_version="3.4",
    )

    chunks, latency_ms = result
    assert isinstance(chunks, list)
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0

    # point3 (score 0.15) should be filtered out because score_threshold = 0.20
    assert len(chunks) == 2

    # Ordering: point2 (0.92) must come before point1 (0.35)
    assert chunks[0].chunk_id == "uuid-2"
    assert chunks[0].score == 0.92
    assert chunks[0].section_title == "Section 2"
    assert chunks[0].doc_url == "https://docs.spring.io/2"
    assert chunks[0].text == "High score content"

    assert chunks[1].chunk_id == "uuid-1"
    assert chunks[1].score == 0.35
