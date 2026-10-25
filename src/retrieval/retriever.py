"""Hybrid vector retrieval engine with metadata filtering and latency tracking."""

import time

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
from src.retrieval.models import RetrievalResult, RetrievedChunk


class DocumentRetriever:
    """Retrieves relevant document chunks from Qdrant using dense and hybrid search."""

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

    def build_version_filter(self, spring_version: str | None) -> Filter | None:
        """Construct Qdrant Filter for matching spring_version."""
        if not spring_version:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="spring_version",
                    match=MatchValue(value=spring_version),
                )
            ]
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        spring_version: str | None = None,
    ) -> RetrievalResult:
        """Retrieve top_k chunks matching query and spring_version filter, sorted by score.

        Returns a RetrievalResult (accessible as namedtuple or unpacked as (chunks, latency_ms)).
        """
        start_time = time.perf_counter()
        target_version = spring_version or settings.spring_version
        filter_condition = self.build_version_filter(target_version)

        dense_vector, sparse_vector = self.embedder.embed_text(query)

        prefetch_queries = [
            Prefetch(
                query=dense_vector,
                using="dense",
                filter=filter_condition,
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using="sparse",
                filter=filter_condition,
                limit=top_k * 2,
            ),
        ]

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        except Exception:
            # Fallback to direct dense search if hybrid query is unavailable
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=filter_condition,
                limit=top_k,
                with_payload=True,
            ).points

        retrieved_chunks: list[RetrievedChunk] = []
        for point in points:
            score = float(point.score or 0.0)
            if score_threshold > 0.0 and score < score_threshold:
                continue

            payload = point.payload or {}
            chunk = RetrievedChunk(
                chunk_id=str(point.id),
                text=payload.get("content", ""),
                doc_url=payload.get("doc_url", ""),
                section_title=payload.get("section_title", "Documentação"),
                spring_version=payload.get("spring_version", target_version or "3.4"),
                score=round(score, 5),
                module=payload.get("module", "core"),
                has_code=bool(payload.get("has_code", False)),
            )
            retrieved_chunks.append(chunk)

        # Ensure strict descending score order
        retrieved_chunks.sort(key=lambda c: c.score, reverse=True)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return RetrievalResult(
            chunks=retrieved_chunks,
            retrieval_latency_ms=elapsed_ms,
        )
