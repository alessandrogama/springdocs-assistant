"""Unit tests for deterministic hashing and UUIDv5 generation."""

import uuid

from src.ingestion.hasher import compute_content_hash, generate_chunk_id


def test_generate_chunk_id_is_deterministic() -> None:
    """Ensure identical inputs always yield the exact same UUIDv5 string."""
    id1 = generate_chunk_id(
        spring_version="3.4",
        doc_url="https://docs.spring.io/spring-boot/reference/web.html",
        section_title="Building REST Services",
        chunk_index=0,
    )
    id2 = generate_chunk_id(
        spring_version="3.4",
        doc_url="https://docs.spring.io/spring-boot/reference/web.html",
        section_title="Building REST Services",
        chunk_index=0,
    )
    assert id1 == id2
    # Verify it is a valid UUID
    parsed = uuid.UUID(id1)
    assert parsed.version == 5


def test_generate_chunk_id_differentiates_parameters() -> None:
    """Ensure altering any parameter alters the resulting UUID."""
    base = generate_chunk_id("3.4", "https://url.com", "Section A", 0)
    diff_ver = generate_chunk_id("3.3", "https://url.com", "Section A", 0)
    diff_url = generate_chunk_id("3.4", "https://other.com", "Section A", 0)
    diff_sec = generate_chunk_id("3.4", "https://url.com", "Section B", 0)
    diff_idx = generate_chunk_id("3.4", "https://url.com", "Section A", 1)

    assert len({base, diff_ver, diff_url, diff_sec, diff_idx}) == 5


def test_compute_content_hash_consistency_and_normalization() -> None:
    """Test SHA-256 content hash calculation and whitespace normalization."""
    text_raw = "  @RestController\npublic class HelloController {}  \n"
    text_clean = "@RestController\npublic class HelloController {}"

    hash_raw = compute_content_hash(text_raw)
    hash_clean = compute_content_hash(text_clean)

    assert len(hash_raw) == 64
    assert hash_raw == hash_clean
