"""Unit tests for semantic chunker and code preservation."""

from src.ingestion.chunker import SemanticCodePreservingChunker
from src.ingestion.parser import ParsedSection


def test_chunker_never_breaks_java_code_blocks() -> None:
    """Ensure multi-line Java code block remains completely intact in chunking."""
    large_java_code = """```java
package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @GetMapping("/greeting")
    public String greeting(@RequestParam(defaultValue = "World") String name) {
        return "Hello, " + name + "!";
    }
}
```"""

    text_intro = "This section demonstrates how to create a complete Spring Boot REST application from scratch."
    full_content = f"{text_intro}\n\n{large_java_code}"

    section = ParsedSection(title="REST Application Guide", content=full_content)
    chunker = SemanticCodePreservingChunker(target_chunk_chars=500)

    chunks = chunker.chunk_section(
        section=section,
        doc_url="https://docs.spring.io/spring-boot/reference/web.html",
        spring_version="3.4",
        module="web",
    )

    assert len(chunks) >= 1
    # Check that the code block exists in its entirety in at least one chunk
    chunk_with_code = [c for c in chunks if c.has_code]
    assert len(chunk_with_code) >= 1
    assert large_java_code.strip() in chunk_with_code[0].content


def test_chunker_never_breaks_yaml_blocks() -> None:
    """Ensure YAML configuration blocks remain intact."""
    yaml_code = """```yaml
spring:
  application:
    name: springdocs-assistant
  datasource:
    url: jdbc:postgresql://localhost:5432/mydb
    username: postgres
    password: secret
    hikari:
      maximum-pool-size: 10
server:
  port: 8080
```"""
    content = f"Configuration details below:\n\n{yaml_code}\n\nEnsure database is reachable."
    section = ParsedSection(title="Datasource Configuration", content=content)
    chunker = SemanticCodePreservingChunker(target_chunk_chars=300)

    chunks = chunker.chunk_section(
        section=section,
        doc_url="https://docs.spring.io/spring-boot/reference/data.html",
        spring_version="3.4",
        module="data",
    )

    code_chunks = [c for c in chunks if c.has_code]
    assert len(code_chunks) >= 1
    assert "maximum-pool-size: 10" in code_chunks[0].content
    assert "spring:" in code_chunks[0].content


def test_chunker_populates_mandatory_metadata() -> None:
    """Verify that all required metadata fields are accurately set."""
    section = ParsedSection(
        title="Security Filters",
        content="Spring Security configures SecurityFilterChain by default.",
        anchor="sec-filters",
    )
    chunker = SemanticCodePreservingChunker()

    chunks = chunker.chunk_section(
        section=section,
        doc_url="https://docs.spring.io/spring-boot/reference/security.html",
        spring_version="3.4",
        module="security",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_url == "https://docs.spring.io/spring-boot/reference/security.html"
    assert chunk.spring_version == "3.4"
    assert chunk.section_title == "Security Filters"
    assert chunk.module == "security"
    assert chunk.chunk_index == 0
    assert len(chunk.content_hash) == 64
    assert chunk.id is not None
