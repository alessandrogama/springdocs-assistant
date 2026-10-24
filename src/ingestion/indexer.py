"""Idempotent vector indexing into Qdrant vector database."""

from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.models import DocumentChunk, IngestionStats


class QdrantIndexer:
    """Manages collection lifecycle and idempotent indexing into Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: DocumentEmbedder | None = None,
        collection_name: str | None = None,
        dense_dim: int | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        self.dense_dim = dense_dim or settings.embedding_dim
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30.0,
        )
        self.embedder = embedder or DocumentEmbedder(
            dense_model_name=settings.embedding_model,
        )

    def ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with dense and sparse vector configurations."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            # Create payload indices for fast filtered queries
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="spring_version",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="module",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def index_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 32,
    ) -> IngestionStats:
        """Idempotently index chunks: only compute embeddings and upsert when content_hash changed."""
        self.ensure_collection()
        stats = IngestionStats(
            total_chunks=len(chunks),
            chunks_with_code=sum(1 for c in chunks if c.has_code),
        )

        if not chunks:
            return stats

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_ids = [c.id for c in batch]

            # Fetch existing records to check content_hash
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=batch_ids,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            existing_hashes = {
                str(p.id): (p.payload or {}).get("content_hash") for p in existing_points
            }

            chunks_to_upsert: list[DocumentChunk] = []
            for chunk in batch:
                existing_hash = existing_hashes.get(str(chunk.id))
                if existing_hash == chunk.content_hash:
                    stats.unchanged_chunks += 1
                else:
                    chunks_to_upsert.append(chunk)

            if not chunks_to_upsert:
                continue

            # Compute embeddings for new or modified chunks
            texts = [c.content for c in chunks_to_upsert]
            dense_vectors = self.embedder.embed_dense_batch(texts)
            sparse_vectors = self.embedder.embed_sparse_batch(texts)

            points_to_insert: list[PointStruct] = []
            for chunk, dense_vec, sparse_vec in zip(
                chunks_to_upsert,
                dense_vectors,
                sparse_vectors,
                strict=True,
            ):
                points_to_insert.append(
                    PointStruct(
                        id=chunk.id,
                        vector={
                            "dense": dense_vec,
                            "sparse": SparseVector(
                                indices=sparse_vec.indices,
                                values=sparse_vec.values,
                            ),
                        },
                        payload=chunk.model_dump(),
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_insert,
            )
            stats.new_or_updated_chunks += len(points_to_insert)

        return stats
