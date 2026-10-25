"""Direct asynchronous client communicating with Ollama runtime."""

import time
from collections.abc import Sequence

import httpx

from src.config import settings
from src.generation.models import (
    GenerationResult,
    OllamaClientError,
    OllamaConnectionError,
    OllamaTimeoutError,
)
from src.generation.prompts import (
    STANDARD_ABSTENTION_PHRASE,
    build_prompts,
    extract_and_validate_citations,
    is_abstention_text,
)
from src.retrieval.models import RetrievedChunk


class OllamaGenerator:
    """Asynchronous generation client communicating with Ollama /api/chat endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float | None = None,
        default_temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.default_model = default_model or settings.llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.default_temperature = default_temperature

    async def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk] | None = None,
        use_rag: bool = True,
        model_name: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        """Generate response with grounding, abstention short-circuit, and citation validation.

        When use_rag is True and 0 chunks are provided, triggers an immediate short-circuit,
        guaranteeing the standard abstention phrase without unnecessary inference latency.
        """
        target_model = model_name or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        timeout = timeout_seconds or self.timeout_seconds
        chunks_list = list(chunks) if chunks is not None else []

        # Short-circuit if RAG is requested but no chunks exist
        if use_rag and len(chunks_list) == 0:
            return GenerationResult(
                answer=STANDARD_ABSTENTION_PHRASE,
                citations=[],
                abstained=True,
                model_used=target_model,
                generation_latency_ms=0.0,
            )

        system_prompt, user_prompt = build_prompts(
            query=query,
            chunks=chunks_list,
            use_rag=use_rag,
        )

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temp,
                "num_predict": 384,
            },
            "stream": False,
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Timeout of {timeout}s exceeded while querying Ollama at {self.base_url}"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama service at {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {exc}") from exc

        generation_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        raw_answer = data.get("message", {}).get("content", "").strip()

        abstained = is_abstention_text(raw_answer)
        if abstained:
            validated_citations: list[str] = []
        else:
            # Validates that all citations refer strictly to URLs present in chunks_list
            validated_citations = extract_and_validate_citations(raw_answer, chunks_list)

        return GenerationResult(
            answer=raw_answer,
            citations=validated_citations,
            abstained=abstained,
            model_used=target_model,
            generation_latency_ms=generation_latency_ms,
        )
