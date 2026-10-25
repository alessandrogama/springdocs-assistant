"""Direct LLM generation client for Ollama runtime."""

import time

import httpx

from src.config import settings


class OllamaGenerator:
    """Async client communicating with local Ollama service using httpx."""

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.default_model = default_model or settings.llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, float]:
        """Generate response via Ollama /api/chat endpoint and return (answer, latency_ms)."""
        target_model = model_name or self.default_model
        temp = temperature if temperature is not None else settings.llm_temperature

        start_time = time.perf_counter()

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

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        generation_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        message_content = data.get("message", {}).get("content", "").strip()

        return message_content, generation_latency_ms

    async def close(self) -> None:
        """Close open resources (no-op when using per-request clients)."""
        pass

