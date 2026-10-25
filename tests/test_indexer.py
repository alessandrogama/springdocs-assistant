"""Unit tests for idempotent Qdrant indexing using mocks."""

from unittest.mock import MagicMock

from qdrant_client.models import CollectionsResponse, PointStruct

from src.ingestion.hasher import compute_content_hash, generate_chunk_id
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.models import DocumentChunk


def create_sample_chunk(
    index: int = 0,
    content: str = "Test Spring Boot content",
    spring_version: str = "3.4.0",
    section: str = "Test Section",
) -> DocumentChunk:
    """Helper to generate a valid DocumentChunk with deterministic ID and hash."""
    doc_url = "https://docs.spring.io/test.html"
    c_hash = compute_content_hash(content)
    c_id = generate_chunk_id(spring_version, doc_url, section, index)
    return DocumentChunk(
        id=c_id,
        doc_url=doc_url,
        spring_version=spring_version,
        section_title=section,
        module="core",
        chunk_index=index,
        content=content,
        content_hash=c_hash,
        has_code=False,
    )


def test_ensure_collection_creates_when_missing() -> None:
    """Validate ensure_collection creates collection and payload indices when collection does not exist."""
    mock_client = MagicMock()
    mock_collections_resp = MagicMock(spec=CollectionsResponse)
    mock_collections_resp.collections = []
    mock_client.get_collections.return_value = mock_collections_resp

    mock_embedder = MagicMock()

    indexer = QdrantIndexer(
        client=mock_client,
        embedder=mock_embedder,
        collection_name="test_collection",
        dense_dim=384,
    )

    indexer.ensure_collection()

    mock_client.create_collection.assert_called_once()
    assert mock_client.create_payload_index.call_count == 3


def test_ensure_collection_skips_creation_when_already_exists() -> None:
    """Validate ensure_collection does not recreate collection if already present."""
    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_coll.name = "test_collection"
    mock_collections_resp = MagicMock(spec=CollectionsResponse)
    mock_collections_resp.collections = [mock_coll]
    mock_client.get_collections.return_value = mock_collections_resp

    indexer = QdrantIndexer(
        client=mock_client,
        embedder=MagicMock(),
        collection_name="test_collection",
    )

    indexer.ensure_collection()
    mock_client.create_collection.assert_not_called()


def test_index_chunks_empty() -> None:
    """Validate indexing an empty chunk list returns zeroed stats."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    indexer = QdrantIndexer(client=mock_client, embedder=MagicMock())
    stats = indexer.index_chunks([])

    assert stats.total_chunks == 0
    assert stats.inserted == 0
    assert stats.updated == 0
    assert stats.skipped == 0


def test_index_chunks_idempotence_inserted_updated_skipped() -> None:
    """Validate idempotent indexing: inserts new, updates modified, skips unchanged."""
    chunk1 = create_sample_chunk(0, "Content A")
    chunk2 = create_sample_chunk(1, "Content B (modified)")
    chunk3 = create_sample_chunk(2, "Content C (unchanged)")

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    # Existing records in Qdrant:
    # - chunk1 is not present
    # - chunk2 has an older hash
    # - chunk3 has the exact same hash
    existing_p2 = MagicMock()
    existing_p2.id = chunk2.id
    existing_p2.payload = {"content_hash": "old_hash_123"}

    existing_p3 = MagicMock()
    existing_p3.id = chunk3.id
    existing_p3.payload = {"content_hash": chunk3.content_hash}

    mock_client.retrieve.return_value = [existing_p2, existing_p3]

    mock_embedder = MagicMock()
    mock_embedder.embed_dense_batch.return_value = [[0.1] * 384, [0.2] * 384]

    mock_sparse1 = MagicMock()
    mock_sparse1.indices = [1, 2]
    mock_sparse1.values = [0.5, 0.8]
    mock_sparse2 = MagicMock()
    mock_sparse2.indices = [3, 4]
    mock_sparse2.values = [0.9, 0.4]
    mock_embedder.embed_sparse_batch.return_value = [mock_sparse1, mock_sparse2]

    indexer = QdrantIndexer(
        client=mock_client,
        embedder=mock_embedder,
        collection_name="spring_docs_v3_4",
    )

    stats = indexer.index_chunks([chunk1, chunk2, chunk3])

    assert stats.total_chunks == 3
    assert stats.inserted == 1  # chunk1
    assert stats.updated == 1   # chunk2
    assert stats.skipped == 1   # chunk3

    # Upsert was called only for chunk1 and chunk2 (2 points)
    mock_client.upsert.assert_called_once()
    _, kwargs = mock_client.upsert.call_args
    points: list[PointStruct] = kwargs["points"]
    assert len(points) == 2
    assert points[0].id == chunk1.id
    assert points[1].id == chunk2.id


def test_indexer_payload_index_handles_existing_index_error() -> None:
    """Validate ensure_collection gracefully ignores exceptions when payload index exists."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_payload_index.side_effect = Exception("Index already exists")

    indexer = QdrantIndexer(client=mock_client, embedder=MagicMock())
    # Should not raise exception
    indexer.ensure_collection()
