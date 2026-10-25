"""Document fetcher for loading Spring Boot documentation."""

from collections.abc import Sequence

import httpx

from src.ingestion.curated_docs import SPRING_BOOT_3_4_DOCS
from src.ingestion.models import RawDocument


class DocumentFetcher:
    """Fetches or loads official Spring Boot documentation sources."""

    def __init__(self, default_docs: Sequence[RawDocument] | None = None) -> None:
        self.default_docs = list(default_docs) if default_docs is not None else SPRING_BOOT_3_4_DOCS

    def fetch_all(self) -> list[RawDocument]:
        """Fetch all configured documentation sources."""
        return list(self.default_docs)

    async def fetch_url(
        self, url: str, title: str, module: str, spring_version: str = "3.4"
    ) -> RawDocument:
        """Fetch a single live documentation page via HTTP."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text

        return RawDocument(
            doc_url=url,
            spring_version=spring_version,
            module=module,
            title=title,
            content=content,
        )
