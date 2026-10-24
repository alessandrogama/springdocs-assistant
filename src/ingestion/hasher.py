"""Deterministic hashing and ID generation for idempotent documentation ingestion."""

import hashlib
import uuid

NAMESPACE_SPRINGDOCS: uuid.UUID = uuid.UUID("a7e58df1-b44c-4a3f-8fa2-938d21c7d6e4")


def generate_chunk_id(
    spring_version: str,
    doc_url: str,
    section_title: str,
    chunk_index: int,
    namespace: uuid.UUID = NAMESPACE_SPRINGDOCS,
) -> str:
    """Generate a deterministic UUIDv5 identifier for a documentation chunk.

    The key is composed of: spring_version + doc_url + section_title + chunk_index
    """
    composite_key = f"{spring_version}:{doc_url}:{section_title}:{chunk_index}"
    return str(uuid.uuid5(namespace, composite_key))


def compute_content_hash(content: str) -> str:
    """Compute normalized SHA-256 hash for document chunk content."""
    normalized = content.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
