"""Prompt engineering, grounding templates, and citation validation for SpringDocs Assistant."""

import re
from collections.abc import Sequence

from src.retrieval.models import RetrievedChunk

STANDARD_ABSTENTION_PHRASE: str = (
    "Não encontrei informações suficientes na documentação indexada do Spring Boot "
    "para responder a esta pergunta."
)

SYSTEM_PROMPT_RAG: str = f"""Você é o assistente técnico especializado no ecossistema Spring Boot 3.x.
Sua missão é responder com máxima precisão técnica, clareza e fidelidade ao contexto fornecido.

Diretrizes obrigatórias:
1. EXCLUSIVIDADE DO CONTEXTO: Responda exclusivamente com base nos blocos [FONTE X] fornecidos no contexto documental. Não utilize conhecimento prévio externo para supor regras não declaradas.
2. CITAÇÃO OBRIGATÓRIA: Toda afirmação técnica, recomendação ou exemplo de código DEVE citar a fonte no formato exato:
   [Fonte: Título da Seção (URL)]
   Utilize apenas os títulos e URLs fornecidos no contexto. NUNCA invente ou altere URLs.
3. PRESERVAÇÃO DE CÓDIGO: Mantenha sempre a integridade sintática e a formatação exata de trechos de código Java e configurações YAML/properties.
4. ABSTENÇÃO OBRIGATÓRIA: Se os trechos recuperados estiverem vazios ou não contiverem a resposta exata para a pergunta, recuse-se a adivinhar e responda exatamente com a seguinte frase:
   "{STANDARD_ABSTENTION_PHRASE}"
"""

SYSTEM_PROMPT_BASE: str = """Você é um assistente técnico especialista em Spring Boot 3.x.
Responda a dúvidas técnicas em Português do Brasil com precisão, concisão e código limpo.
"""


def build_prompts(
    query: str,
    chunks: Sequence[RetrievedChunk] | None = None,
    use_rag: bool = True,
) -> tuple[str, str]:
    """Build system and user prompts depending on whether RAG context is enabled."""
    if not use_rag:
        return SYSTEM_PROMPT_BASE, query

    chunks_list = list(chunks) if chunks is not None else []
    if not chunks_list:
        context_str = "Nenhum documento relevante encontrado no índice."
    else:
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks_list, 1):
            source_tag = f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
            context_parts.append(
                f"[FONTE {i}]\n"
                f"Referência: {source_tag}\n"
                f"Módulo: {chunk.module} (Spring Boot {chunk.spring_version})\n"
                f"Conteúdo:\n{chunk.text}\n"
            )
        context_str = "\n".join(context_parts)

    user_prompt = f"""Contexto documental indexado:
{context_str}

Pergunta do desenvolvedor:
{query}

Resposta técnica fundamentada exclusivamente no contexto acima (incluindo citações):"""

    return SYSTEM_PROMPT_RAG, user_prompt


def extract_and_validate_citations(
    text: str,
    context_chunks: Sequence[RetrievedChunk] | None,
) -> list[str]:
    """Extract citations from text and strictly validate against context chunk URLs.

    Guarantees that the generator never admits or invents URLs that were not
    present in the retrieved context chunks.
    """
    if not context_chunks:
        return []

    # Map allowed URLs and canonical section titles from retrieved chunks
    valid_urls_map: dict[str, str] = {
        chunk.doc_url.strip(): chunk.section_title.strip() for chunk in context_chunks
    }

    citation_pattern = re.compile(r"\[Fonte:\s*([^\]]+?)\s*\((https?://[^\)]+)\)\]")
    matches = citation_pattern.findall(text)

    validated_citations: list[str] = []
    seen: set[str] = set()

    for title, url in matches:
        clean_url = url.strip()
        clean_title = title.strip()

        # Reject any URL that does not belong to the retrieved context chunks
        if clean_url in valid_urls_map:
            citation_str = f"[Fonte: {clean_title} ({clean_url})]"
            if citation_str not in seen:
                seen.add(citation_str)
                validated_citations.append(citation_str)

    # Fallback heuristic: If the model mentioned the section title but missed exact citation markdown,
    # match against verified context chunks
    if not validated_citations:
        lower_text = text.lower()
        for chunk in context_chunks:
            if chunk.section_title.lower() in lower_text:
                citation_str = f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
                if citation_str not in seen:
                    seen.add(citation_str)
                    validated_citations.append(citation_str)

    return validated_citations


def is_abstention_text(text: str) -> bool:
    """Check if the text represents safe abstention."""
    normalized = text.strip().lower()
    if STANDARD_ABSTENTION_PHRASE.lower() in normalized:
        return True
    abstention_indicators = [
        "não encontrei informações suficientes",
        "não contém informações",
        "não foram encontradas informações",
        "não há informações suficientes",
        "fora de escopo",
    ]
    return any(ind in normalized for ind in abstention_indicators)
