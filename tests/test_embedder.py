"""Unit tests for DocumentEmbedder and SparseVectorData using mocks."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.ingestion.embedder import DocumentEmbedder, SparseVectorData


def test_sparse_vector_data_structure() -> None:
    """Validate SparseVectorData stores indices and values."""
    sparse = SparseVectorData(indices=[1, 5, 10], values=[0.2, 0.5, 0.9])
    assert sparse.indices == [1, 5, 10]
    assert sparse.values == [0.2, 0.5, 0.9]


def test_document_embedder_alias_mapping() -> None:
    """Validate DocumentEmbedder maps intfloat/multilingual-e5-small to supported FastEmbed model."""
    with (
        patch("src.ingestion.embedder.TextEmbedding") as mock_dense_cls,
        patch("src.ingestion.embedder.SparseTextEmbedding") as mock_sparse_cls,
    ):
        embedder = DocumentEmbedder(dense_model_name="intfloat/multilingual-e5-small")
        mock_dense_cls.assert_called_once_with(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        mock_sparse_cls.assert_called_once_with(model_name="Qdrant/bm25")
        assert embedder.dense_model_name == "intfloat/multilingual-e5-small"


def test_document_embedder_batch_and_single_text() -> None:
    """Validate embed_dense_batch, embed_sparse_batch and embed_text methods."""
    with (
        patch("src.ingestion.embedder.TextEmbedding") as mock_dense_cls,
        patch("src.ingestion.embedder.SparseTextEmbedding") as mock_sparse_cls,
    ):
        mock_dense_model = MagicMock()
        mock_dense_model.embed.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_dense_cls.return_value = mock_dense_model

        mock_sparse_res = MagicMock()
        mock_sparse_res.indices = np.array([2, 4])
        mock_sparse_res.values = np.array([0.8, 0.4])
        mock_sparse_model = MagicMock()
        mock_sparse_model.embed.return_value = [mock_sparse_res]
        mock_sparse_cls.return_value = mock_sparse_model

        embedder = DocumentEmbedder()

        # Test batch methods
        dense_batch = embedder.embed_dense_batch(["Test doc"])
        assert len(dense_batch) == 1
        assert dense_batch[0] == [0.1, 0.2, 0.3]

        sparse_batch = embedder.embed_sparse_batch(["Test doc"])
        assert len(sparse_batch) == 1
        assert sparse_batch[0].indices == [2, 4]
        assert sparse_batch[0].values == [0.8, 0.4]

        # Test embed_text
        dense_vec, sparse_vec = embedder.embed_text("Test query")
        assert dense_vec == [0.1, 0.2, 0.3]
        assert sparse_vec.indices == [2, 4]
