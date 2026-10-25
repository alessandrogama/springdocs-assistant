"""Main ingestion and indexing pipeline for SpringDocs Assistant."""

import json
import time
from collections.abc import Sequence
from pathlib import Path

from src.ingestion.chunker import SemanticCodePreservingChunker
from src.ingestion.fetcher import DocumentFetcher
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.models import DocumentChunk, IngestionStats, RawDocument
from src.ingestion.parser import DocumentParser

DEFAULT_PROCESSED_FILE = Path("data/processed/chunks.jsonl")


class IngestionPipeline:
    """Orchestrates fetcher -> parser -> chunker -> indexer pipeline."""

    def __init__(
        self,
        fetcher: DocumentFetcher | None = None,
        parser: DocumentParser | None = None,
        chunker: SemanticCodePreservingChunker | None = None,
        indexer: QdrantIndexer | None = None,
        processed_path: Path = DEFAULT_PROCESSED_FILE,
    ) -> None:
        self.fetcher = fetcher or DocumentFetcher()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or SemanticCodePreservingChunker()
        self.indexer = indexer or QdrantIndexer()
        self.processed_path = processed_path

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

    def save_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Serialize chunks into JSON Lines format at data/processed/chunks.jsonl."""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    def load_chunks(self) -> list[DocumentChunk]:
        """Load chunks from data/processed/chunks.jsonl."""
        if not self.processed_path.exists():
            return []
        chunks: list[DocumentChunk] = []
        with open(self.processed_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    data = json.loads(line_str)
                    chunks.append(DocumentChunk(**data))
        return chunks

    def run(self, force_fetch: bool = False) -> IngestionStats:
        """Execute full ingestion pipeline.

        Encadeia fetcher -> parser -> chunker -> indexer.
        If data/processed/chunks.jsonl does not exist or force_fetch is True,
        fetches docs, chunks them, and saves to chunks.jsonl before indexing.
        """
        start_time = time.perf_counter()

        if force_fetch or not self.processed_path.exists():
            print("[*] Fetching documentation sources...")
            docs = self.fetcher.fetch_all()
            print(f"[*] Fetched {len(docs)} documents. Parsing and chunking...")
            chunks = self.process_documents(docs)
            print(f"[*] Extracted {len(chunks)} chunks. Saving to {self.processed_path}...")
            self.save_chunks(chunks)
            total_docs = len(docs)
        else:
            print(f"[*] Loading pre-processed chunks from {self.processed_path}...")
            chunks = self.load_chunks()
            total_docs = len({c.doc_url for c in chunks})

        print(
            f"[*] Ready to index {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.has_code)} with code)."
        )

        print("[*] Performing idempotent indexing into Qdrant...")
        stats = self.indexer.index_chunks(chunks)
        stats.total_documents = total_docs
        stats.duration_seconds = round(time.perf_counter() - start_time, 3)

        print(
            f"[+] Ingestion complete in {stats.duration_seconds}s: "
            f"inserted={stats.inserted}, updated={stats.updated}, skipped={stats.skipped}."
        )
        return stats


def main() -> None:
    """CLI entrypoint for running ingestion pipeline."""
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
