"""Main ingestion and indexing pipeline for SpringDocs Assistant."""

import time
from collections.abc import Sequence

from src.ingestion.chunker import SemanticCodePreservingChunker
from src.ingestion.curated_docs import SPRING_BOOT_3_4_DOCS
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.models import DocumentChunk, IngestionStats, RawDocument
from src.ingestion.parser import DocumentParser


class IngestionPipeline:
    """Orchestrates parsing, code-preserving chunking, and idempotent indexing."""

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: SemanticCodePreservingChunker | None = None,
        indexer: QdrantIndexer | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.chunker = chunker or SemanticCodePreservingChunker()
        self.indexer = indexer or QdrantIndexer()

    def process_documents(self, documents: Sequence[RawDocument]) -> list[DocumentChunk]:
        """Convert a sequence of raw documents into code-preserved chunks with metadata."""
        all_chunks: list[DocumentChunk] = []

        for doc in documents:
            if "<html" in doc.content.lower() or "<article" in doc.content.lower():
                sections = self.parser.parse_html(doc.content, default_title=doc.title)
            else:
                sections = self.parser.parse_markdown(doc.content, default_title=doc.title)

            chunks = self.chunker.chunk_document(
                sections=sections,
                doc_url=doc.doc_url,
                spring_version=doc.spring_version,
                module=doc.module,
            )
            all_chunks.extend(chunks)

        return all_chunks

    def run(self, documents: Sequence[RawDocument] | None = None) -> IngestionStats:
        """Execute full ingestion pipeline for provided or default Phase 1 documents."""
        docs = documents if documents is not None else SPRING_BOOT_3_4_DOCS
        start_time = time.perf_counter()

        print(f"[*] Starting ingestion for {len(docs)} documentation pages...")
        chunks = self.process_documents(docs)
        print(
            f"[*] Extracted {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.has_code)} containing code blocks)."
        )

        print("[*] Indexing chunks idempotently into Qdrant...")
        stats = self.indexer.index_chunks(chunks)
        stats.total_documents = len(docs)
        stats.duration_seconds = round(time.perf_counter() - start_time, 3)

        print(
            f"[+] Ingestion complete in {stats.duration_seconds}s. "
            f"New/Updated: {stats.new_or_updated_chunks}, "
            f"Unchanged: {stats.unchanged_chunks}."
        )
        return stats


def main() -> None:
    """CLI entrypoint for running ingestion pipeline."""
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
