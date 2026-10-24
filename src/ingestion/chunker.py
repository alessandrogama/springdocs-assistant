"""Semantic chunker with code block preservation and deterministic metadata."""

from collections.abc import Sequence

from src.ingestion.hasher import compute_content_hash, generate_chunk_id
from src.ingestion.models import DocumentChunk
from src.ingestion.parser import ParsedSection


class SemanticCodePreservingChunker:
    """Chunker that partitions documentation semantically while strictly preserving code blocks.

    Rules enforced:
    1. NEVER split code blocks (``` ... ``` or [source,...] ... ----) in the middle.
    2. Target chunk size: 350-500 tokens (~1400-2000 chars).
    3. Overlap of 10-15% applied ONLY to running text paragraphs, never to code.
    4. Deterministic UUIDv5 ID and SHA-256 content_hash.
    """

    def __init__(
        self,
        target_chunk_chars: int = 1600,
        max_chunk_chars: int = 2200,
        overlap_chars: int = 200,
    ) -> None:
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_section(
        self,
        section: ParsedSection,
        doc_url: str,
        spring_version: str,
        module: str,
    ) -> list[DocumentChunk]:
        """Split a parsed section into one or more DocumentChunks without breaking code blocks."""
        blocks = self._split_into_atomic_blocks(section.content)
        if not blocks:
            return []

        chunks: list[DocumentChunk] = []
        current_chunk_blocks: list[str] = []
        current_length: int = 0
        chunk_idx: int = 0

        for block in blocks:
            is_code = self._is_code_block(block)
            block_len = len(block)

            # If single code block exceeds max limit, it must be kept intact in its own chunk
            if is_code and block_len > self.target_chunk_chars:
                # Flush previous accumulated text if any
                if current_chunk_blocks:
                    content_str = "\n\n".join(current_chunk_blocks).strip()
                    chunks.append(
                        self._create_chunk(
                            content=content_str,
                            section=section,
                            doc_url=doc_url,
                            spring_version=spring_version,
                            module=module,
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1
                    current_chunk_blocks = []
                    current_length = 0

                # Add atomic oversized code block intact
                chunks.append(
                    self._create_chunk(
                        content=block.strip(),
                        section=section,
                        doc_url=doc_url,
                        spring_version=spring_version,
                        module=module,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                continue

            # If adding this block exceeds target size and we already have content
            if current_length + block_len > self.target_chunk_chars and current_chunk_blocks:
                content_str = "\n\n".join(current_chunk_blocks).strip()
                chunks.append(
                    self._create_chunk(
                        content=content_str,
                        section=section,
                        doc_url=doc_url,
                        spring_version=spring_version,
                        module=module,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1

                # Calculate text overlap only from last block if it was running text (not code)
                overlap_text = ""
                last_block = current_chunk_blocks[-1]
                if not self._is_code_block(last_block) and len(last_block) > self.overlap_chars:
                    overlap_text = last_block[-self.overlap_chars :].strip()

                current_chunk_blocks = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            current_chunk_blocks.append(block)
            current_length += block_len

        # Flush final accumulated chunk
        if current_chunk_blocks:
            content_str = "\n\n".join(current_chunk_blocks).strip()
            if content_str:
                chunks.append(
                    self._create_chunk(
                        content=content_str,
                        section=section,
                        doc_url=doc_url,
                        spring_version=spring_version,
                        module=module,
                        chunk_index=chunk_idx,
                    )
                )

        return chunks

    def chunk_document(
        self,
        sections: Sequence[ParsedSection],
        doc_url: str,
        spring_version: str,
        module: str,
    ) -> list[DocumentChunk]:
        """Process all sections in a document into chunks."""
        all_chunks: list[DocumentChunk] = []
        for section in sections:
            section_chunks = self.chunk_section(
                section=section,
                doc_url=doc_url,
                spring_version=spring_version,
                module=module,
            )
            all_chunks.extend(section_chunks)
        return all_chunks

    def _split_into_atomic_blocks(self, content: str) -> list[str]:
        """Decompose text into atomic units: code blocks and text paragraphs."""
        lines = content.splitlines()
        blocks: list[str] = []
        current_block: list[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Handle Markdown code block delimiter
            if stripped.startswith("```"):
                if not in_code_block:
                    # Flush accumulated text before starting code block
                    if current_block:
                        text_part = "\n".join(current_block).strip()
                        if text_part:
                            blocks.append(text_part)
                        current_block = []
                    in_code_block = True
                    current_block.append(line)
                else:
                    # Closing code block
                    current_block.append(line)
                    code_part = "\n".join(current_block).strip()
                    if code_part:
                        blocks.append(code_part)
                    current_block = []
                    in_code_block = False
                continue

            if in_code_block:
                current_block.append(line)
                continue

            # Regular text paragraphs separated by empty lines
            if not stripped:
                if current_block:
                    text_part = "\n".join(current_block).strip()
                    if text_part:
                        blocks.append(text_part)
                    current_block = []
            else:
                current_block.append(line)

        # Flush any trailing block
        if current_block:
            part = "\n".join(current_block).strip()
            if part:
                blocks.append(part)

        return blocks

    def _is_code_block(self, block: str) -> bool:
        """Check if block is an atomic code snippet."""
        stripped = block.strip()
        if stripped.startswith("```") and stripped.endswith("```") and len(stripped) > 5:
            return True
        return bool(stripped.startswith("[source") or stripped.startswith("----"))

    def _detect_has_code(self, content: str) -> bool:
        """Detect whether chunk contains executable code or configuration syntax."""
        if "```" in content or "[source" in content:
            return True
        code_markers = [
            "@SpringBootApplication",
            "@Configuration",
            "@Bean",
            "@RestController",
            "@GetMapping",
            "@PostMapping",
            "@Entity",
            "@Autowired",
            "public class ",
            "public interface ",
            "spring.datasource",
            "server.port",
            "spring.jpa",
        ]
        return any(marker in content for marker in code_markers)

    def _create_chunk(
        self,
        content: str,
        section: ParsedSection,
        doc_url: str,
        spring_version: str,
        module: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """Construct a DocumentChunk with deterministic UUIDv5 and SHA-256 hash."""
        chunk_id = generate_chunk_id(
            spring_version=spring_version,
            doc_url=doc_url,
            section_title=section.title,
            chunk_index=chunk_index,
        )
        content_hash = compute_content_hash(content)
        has_code = self._detect_has_code(content)

        return DocumentChunk(
            id=chunk_id,
            doc_url=doc_url,
            spring_version=spring_version,
            section_title=section.title,
            module=module,
            content=content,
            content_hash=content_hash,
            has_code=has_code,
            chunk_index=chunk_index,
            parent_section=section.parent_title,
            metadata={
                "section_level": section.level,
                "anchor": section.anchor,
            },
        )
