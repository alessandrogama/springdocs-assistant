"""Dense and sparse embedding generation using FastEmbed."""

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding


@dataclass
class SparseVectorData:
    """Sparse vector representation with indices and values."""

    indices: list[int]
    values: list[float]


class DocumentEmbedder:
    """Generates dense vectors and BM25 sparse vectors for hybrid retrieval."""

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
    ) -> None:
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self._dense_model = TextEmbedding(model_name=dense_model_name)
        self._sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a batch of texts."""
        embeddings = list(self._dense_model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_sparse_batch(self, texts: list[str]) -> list[SparseVectorData]:
        """Generate sparse lexical vectors (BM25) for a batch of texts."""
        sparse_results = list(self._sparse_model.embed(texts))
        return [
            SparseVectorData(
                indices=res.indices.tolist(),
                values=res.values.tolist(),
            )
            for res in sparse_results
        ]

    def embed_text(self, text: str) -> tuple[list[float], SparseVectorData]:
        """Generate both dense and sparse vectors for a single query or chunk."""
        dense = self.embed_dense_batch([text])[0]
        sparse = self.embed_sparse_batch([text])[0]
        return dense, sparse
