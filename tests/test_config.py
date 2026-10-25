"""Unit tests for configuration loading and default validation."""

import pytest
from pydantic import ValidationError


def test_settings_default_values_without_env_file() -> None:
    """Validate that settings can be imported and load all default values without .env."""
    from src.config import Settings, settings

    # Verify global instance
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.collection_name == "spring_docs_v3_4"
    assert settings.spring_version == "3.4.0"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.embedding_dim == 384
    assert settings.llm_model == "qwen2.5:3b"
    assert settings.llm_temperature == 0.1
    assert settings.top_k == 5
    assert settings.score_threshold == 0.35
    assert settings.rerank_enabled is False

    # Verify fresh instance behaves identically
    custom = Settings()
    assert custom.qdrant_url == "http://localhost:6333"
    assert custom.collection_name == "spring_docs_v3_4"


def test_settings_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that environment variables properly override default configuration values."""
    from src.config import Settings

    monkeypatch.setenv("QDRANT_URL", "http://custom-qdrant:6333")
    monkeypatch.setenv("COLLECTION_NAME", "custom_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.5")

    overridden = Settings(_env_file=None)
    assert overridden.qdrant_url == "http://custom-qdrant:6333"
    assert overridden.collection_name == "custom_collection"
    assert overridden.embedding_model == "BAAI/bge-m3"
    assert overridden.embedding_dim == 1024
    assert overridden.top_k == 10
    assert overridden.score_threshold == 0.5


def test_settings_validation_constraints() -> None:
    """Validate that invalid parameters raise Pydantic validation errors."""
    from src.config import Settings

    with pytest.raises(ValidationError):
        # top_k must be > 0
        Settings(top_k=0)

    with pytest.raises(ValidationError):
        # score_threshold must be between 0.0 and 1.0
        Settings(score_threshold=1.5)
