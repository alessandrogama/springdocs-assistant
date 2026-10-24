"""Hybrid vector and lexical retriever using Qdrant and RRF."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from src.config import settings
from src.ingestion.embedder import DocumentEmbedder
from src.rag.models import RetrievedChunk


class HybridRetriever:
    """Combines dense semantic search and BM25 sparse lexical search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: DocumentEmbedder | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30.0,
        )
        self.embedder = embedder or DocumentEmbedder(
            dense_model_name=settings.embedding_model,
        )

    def retrieve(
        self,
        query: str,
        spring_version: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Execute hybrid search using RRF and return relevant chunks."""
        target_version = spring_version or settings.spring_version
        limit = top_k or settings.top_k

        dense_vector, sparse_vector = self.embedder.embed_text(query)

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="spring_version",
                    match=MatchValue(value=target_version),
                )
            ]
        )

        prefetch_queries = [
            Prefetch(
                query=dense_vector,
                using="dense",
                filter=filter_condition,
                limit=limit * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using="sparse",
                filter=filter_condition,
                limit=limit * 2,
            ),
        ]

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        retrieved: list[RetrievedChunk] = []
        for point in response.points:
            score = float(point.score or 0.0)
            if score_threshold is not None and score < score_threshold:
                continue

            payload = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    id=str(point.id),
                    content=payload.get("content", ""),
                    score=round(score, 5),
                    doc_url=payload.get("doc_url", ""),
                    section_title=payload.get("section_title", "Documentação"),
                    module=payload.get("module", "core"),
                    spring_version=payload.get("spring_version", target_version),
                    has_code=bool(payload.get("has_code", False)),
                )
            )

        return retrieved
