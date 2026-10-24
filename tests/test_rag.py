"""Unit tests for RAG prompt engineering, citations, and abstention handling."""

from src.rag.models import RetrievedChunk
from src.rag.prompt import (
    build_rag_prompt,
    extract_citations,
    is_abstention_response,
)


def test_build_rag_prompt_injects_sources_and_rules() -> None:
    """Ensure system prompt and user context contain formatting and reference tags."""
    chunk = RetrievedChunk(
        id="chunk-123",
        content="HikariCP is the default connection pool.",
        score=0.045,
        doc_url="https://docs.spring.io/data.html",
        section_title="Connection Pool Auto-configuration",
        module="data",
        spring_version="3.4",
        has_code=False,
    )

    sys_prompt, user_prompt = build_rag_prompt(
        query="Qual o pool padrão?",
        chunks=[chunk],
    )

    assert "SpringDocs Assistant" in sys_prompt
    assert "Português do Brasil" in sys_prompt
    assert (
        "[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/data.html)]"
        in user_prompt
    )
    assert "HikariCP" in user_prompt


def test_extract_citations_matches_format() -> None:
    """Verify that [Fonte: Title (URL)] citations are properly parsed."""
    text = (
        "No Spring Boot 3.x, use HikariCP [Fonte: SQL Databases (https://docs.spring.io/sql.html)]."
    )
    citations = extract_citations(text, chunks=[])
    assert len(citations) == 1
    assert citations[0] == "[Fonte: SQL Databases (https://docs.spring.io/sql.html)]"


def test_is_abstention_response_detection() -> None:
    """Ensure abstention detection catches standard refusal keywords."""
    abstain_text = "Não encontrei informações suficientes na documentação oficial fornecida do Spring Boot 3.x."
    normal_text = "Para configurar o HikariCP, declare a propriedade spring.datasource."

    assert is_abstention_response(abstain_text) is True
    assert is_abstention_response(normal_text) is False
