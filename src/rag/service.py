"""RAG Orchestration service coordinating retrieval, prompting, and LLM inference."""

import time

from src.config import settings
from src.rag.generator import OllamaGenerator
from src.rag.models import AskRequest, AskResponse, RetrievedChunk
from src.rag.prompt import (
    ABSTENTION_PHRASE,
    build_rag_prompt,
    extract_citations,
    is_abstention_response,
)
from src.rag.retriever import HybridRetriever


class RagService:
    """End-to-end RAG service with telemetry and safe abstention."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: OllamaGenerator | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or OllamaGenerator()

    async def answer_question(self, request: AskRequest) -> AskResponse:
        """Process user question through hybrid retrieval, prompt grounding, and generation."""
        total_start = time.perf_counter()

        # Step 1: Hybrid Retrieval
        retrieval_start = time.perf_counter()
        retrieved_chunks: list[RetrievedChunk] = self.retriever.retrieve(
            query=request.query,
            spring_version=request.spring_version,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # Step 2: Handle empty context or out of scope queries immediately
        if not retrieved_chunks:
            total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
            return AskResponse(
                answer=f"{ABSTENTION_PHRASE} para responder a esta pergunta.",
                citations=[],
                retrieved_chunks=[],
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                model_name=request.model_name or settings.llm_model,
                spring_version=request.spring_version or settings.spring_version,
                is_abstained=True,
            )

        # Step 3: Build strict grounding prompt
        system_prompt, user_prompt = build_rag_prompt(
            query=request.query,
            chunks=retrieved_chunks,
        )

        # Step 4: Generate LLM response
        answer_text, generation_latency_ms = await self.generator.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=request.model_name,
            temperature=request.temperature,
        )

        total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
        abstained = is_abstention_response(answer_text)
        citations = [] if abstained else extract_citations(answer_text, retrieved_chunks)

        return AskResponse(
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            model_name=request.model_name or settings.llm_model,
            spring_version=request.spring_version or settings.spring_version,
            is_abstained=abstained,
        )
