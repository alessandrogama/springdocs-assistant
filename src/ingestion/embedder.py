"""Dense and sparse embedding generation using FastEmbed."""

import logging
from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding

logger = logging.getLogger(__name__)

# FastEmbed model aliases for lightweight cross-lingual embedding compatibility
FASTEMBED_MODEL_ALIASES: dict[str, str] = {
    "intfloat/multilingual-e5-small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


@dataclass
class SparseVectorData:
    """Sparse vector representation with indices and values."""

    indices: list[int]
    values: list[float]


class DocumentEmbedder:
    """Generates dense vectors and BM25 sparse vectors for hybrid retrieval."""

    def __init__(
        self,
        dense_model_name: str = "intfloat/multilingual-e5-small",
        sparse_model_name: str = "Qdrant/bm25",
    ) -> None:
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

        resolved_dense_model = FASTEMBED_MODEL_ALIASES.get(dense_model_name, dense_model_name)
        if resolved_dense_model != dense_model_name:
            logger.info(
                "Mapping embedding model '%s' to FastEmbed-compatible model '%s'",
                dense_model_name,
                resolved_dense_model,
            )

        self._dense_model = TextEmbedding(model_name=resolved_dense_model)
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
