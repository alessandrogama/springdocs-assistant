"""Prompt engineering and strict grounding builder for SpringDocs Assistant."""

import re

from src.rag.models import RetrievedChunk

SYSTEM_PROMPT = """Você é o SpringDocs Assistant, um especialista de alta precisão técnica em Spring Boot 3.x.
Sua missão é responder perguntas técnicas com base ESTRITAMENTE no contexto documental fornecido.

Diretrizes obrigatórias:
1. IDIOMA: Responda sempre em Português do Brasil (pt-BR) técnico, claro e direto ao ponto.
2. CITAÇÃO OBRIGATÓRIA: Para cada afirmação técnica ou exemplo apresentado, cite a fonte exata no formato:
   [Fonte: Título da Seção (URL)]
3. CÓDIGO ÍNTEGRO: Mantenha sempre a integridade sintática de blocos de código Java, configurações YAML ou propriedades (`application.properties`). Nunca invente ou altere assinaturas de métodos.
4. ABSTENÇÃO SEGURA: Se a pergunta NÃO puder ser respondida com as informações presentes no contexto fornecido (ou se for fora de escopo), você DEVE responder estritamente:
   "Não encontrei informações suficientes na documentação oficial fornecida do Spring Boot 3.x para responder a esta pergunta."
   NUNCA utilize conhecimento externo ou alucine quando o contexto for insuficiente.
"""

ABSTENTION_PHRASE = (
    "Não encontrei informações suficientes na documentação oficial fornecida do Spring Boot 3.x"
)


def build_rag_prompt(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Construct system prompt and user context prompt."""
    if not chunks:
        context_str = "Nenhum documento relevante encontrado na base vetorial."
    else:
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source_tag = f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
            context_parts.append(
                f"--- DOCUMENTO [{i}] ---\n"
                f"Referência: {source_tag}\n"
                f"Módulo: {chunk.module} (Spring Boot {chunk.spring_version})\n"
                f"Conteúdo:\n{chunk.content}\n"
            )
        context_str = "\n".join(context_parts)

    user_prompt = f"""Contexto da documentação oficial:
{context_str}

Pergunta do desenvolvedor:
{query}

Resposta técnica baseada estritamente no contexto acima (com citações de fontes):"""

    return SYSTEM_PROMPT, user_prompt


def extract_citations(text: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Extract and validate citations from text matching [Fonte: ... (...)]."""
    citation_pattern = re.compile(r"\[Fonte:\s*([^\]]+?)\s*\((https?://[^\)]+)\)\]")
    matches = citation_pattern.findall(text)

    citations: list[str] = []
    seen: set[str] = set()

    for title, url in matches:
        cit_str = f"[Fonte: {title.strip()} ({url.strip()})]"
        if cit_str not in seen:
            seen.add(cit_str)
            citations.append(cit_str)

    # If the LLM omitted explicit markdown format but context chunks were provided,
    # include valid matched chunk sources
    if not citations and chunks:
        for chunk in chunks:
            if chunk.section_title.lower() in text.lower():
                cit_str = f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
                if cit_str not in seen:
                    seen.add(cit_str)
                    citations.append(cit_str)

    return citations


def is_abstention_response(text: str) -> bool:
    """Check if the answer indicates safe abstention."""
    keywords = [
        "não encontrei",
        "não contém",
        "insuficiente",
        "fora de escopo",
        "não há informações",
    ]
    lower = text.lower()
    return any(kw in lower for kw in keywords)
