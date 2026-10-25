"""Unit tests for ingestion pipeline, fetcher and persistence."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.fetcher import DocumentFetcher
from src.ingestion.models import DocumentChunk, IngestionStats, RawDocument
from src.ingestion.pipeline import IngestionPipeline, main


def test_fetcher_fetches_curated_docs() -> None:
    """Validate DocumentFetcher returns curated docs list."""
    fetcher = DocumentFetcher()
    docs = fetcher.fetch_all()
    assert len(docs) > 0
    assert isinstance(docs[0], RawDocument)
    assert docs[0].spring_version == "3.4.0"


def test_pipeline_process_documents() -> None:
    """Validate process_documents parses and chunks raw documents."""
    doc = RawDocument(
        doc_url="https://docs.spring.io/test.html",
        spring_version="3.4.0",
        module="web",
        title="Test Doc",
        content="# Title\n\n```java\nclass App {}\n```",
    )
    pipeline = IngestionPipeline()
    chunks = pipeline.process_documents([doc])
    assert len(chunks) >= 1
    assert chunks[0].module == "web"
    assert chunks[0].has_code is True


def test_pipeline_save_and_load_chunks(tmp_path: Path) -> None:
    """Validate serialization and deserialization of chunks to JSONL."""
    jsonl_file = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(processed_path=jsonl_file)

    sample_chunks = [
        DocumentChunk(
            id="chunk-1",
            doc_url="https://test.com",
            spring_version="3.4.0",
            section_title="Sec 1",
            module="core",
            chunk_index=0,
            content="Hello world",
            content_hash="abc123hash",
            has_code=False,
        )
    ]

    pipeline.save_chunks(sample_chunks)
    assert jsonl_file.exists()

    loaded = pipeline.load_chunks()
    assert len(loaded) == 1
    assert loaded[0].id == "chunk-1"
    assert loaded[0].content == "Hello world"


def test_pipeline_load_chunks_missing_file_returns_empty(tmp_path: Path) -> None:
    """Validate load_chunks returns empty list when file does not exist."""
    pipeline = IngestionPipeline(processed_path=tmp_path / "missing.jsonl")
    assert pipeline.load_chunks() == []


def test_pipeline_run_force_fetch(tmp_path: Path) -> None:
    """Validate pipeline run with force_fetch calls fetcher and indexer."""
    mock_indexer = MagicMock()
    mock_indexer.index_chunks.return_value = IngestionStats(
        total_documents=1,
        total_chunks=2,
        chunks_with_code=1,
        inserted=2,
        updated=0,
        skipped=0,
    )

    doc = RawDocument(
        doc_url="https://docs.spring.io/test.html",
        spring_version="3.4.0",
        module="core",
        title="Test",
        content="Short content for testing pipeline.",
    )

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_all.return_value = [doc]

    pipeline = IngestionPipeline(
        fetcher=mock_fetcher,
        indexer=mock_indexer,
        processed_path=tmp_path / "chunks.jsonl",
    )

    stats = pipeline.run(force_fetch=True)
    assert stats.inserted == 2
    assert mock_fetcher.fetch_all.called
    assert mock_indexer.index_chunks.called


def test_pipeline_main_cli_execution() -> None:
    """Validate main entrypoint invokes IngestionPipeline.run."""
    with patch("src.ingestion.pipeline.IngestionPipeline") as mock_pipeline_cls:
        mock_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_instance

        with patch("sys.argv", ["pipeline.py", "--force"]):
            main()
            mock_instance.run.assert_called_once_with(force_fetch=True)
