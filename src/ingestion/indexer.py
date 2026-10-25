"""Idempotent vector indexing into Qdrant vector database."""

import logging
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

logger = logging.getLogger(__name__)


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
        """Ensure Qdrant collection exists with dense and sparse vectors, plus payload indexes."""
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

        # Ensure payload indices on spring_version, module, and content_hash
        indexed_fields = ["spring_version", "module", "content_hash"]
        for field in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001
                # Index may already exist
                logger.debug("Payload index for '%s' already exists or skipped: %s", field, exc)

    def index_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 32,
    ) -> IngestionStats:
        """Idempotently index chunks.

        Only compute embeddings and upsert points that are new (inserted) or whose
        content_hash has changed (updated). Chunks with identical hashes are skipped.
        """
        self.ensure_collection()
        stats = IngestionStats(
            total_chunks=len(chunks),
            chunks_with_code=sum(1 for c in chunks if c.has_code),
        )

        if not chunks:
            print(
                f"[indexer] inserted={stats.inserted}, updated={stats.updated}, skipped={stats.skipped}"
            )
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

            to_insert: list[DocumentChunk] = []
            to_update: list[DocumentChunk] = []

            for chunk in batch:
                chunk_id_str = str(chunk.id)
                if chunk_id_str not in existing_hashes:
                    to_insert.append(chunk)
                elif existing_hashes[chunk_id_str] != chunk.content_hash:
                    to_update.append(chunk)
                else:
                    stats.skipped += 1

            chunks_to_upsert = to_insert + to_update
            if not chunks_to_upsert:
                continue

            # Compute embeddings only for new or modified chunks
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
            stats.inserted += len(to_insert)
            stats.updated += len(to_update)

        # Output the exact required log format
        print(
            f"[indexer] inserted={stats.inserted}, updated={stats.updated}, skipped={stats.skipped}"
        )
        return stats
