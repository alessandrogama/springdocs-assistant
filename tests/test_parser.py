"""Unit tests for HTML and Markdown document parsing and cleaning."""

from src.ingestion.parser import DocumentParser


def test_parse_html_strips_unwanted_tags_and_formats_code() -> None:
    """Test that scripts, nav, and footers are stripped while Java pre blocks become markdown."""
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head><title>Spring Boot Reference</title></head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <header>Header content</header>
        <article>
            <h1 id="intro">Spring Boot Overview</h1>
            <p>Spring Boot makes it easy to create stand-alone production applications.</p>
            <script>alert('malicious')</script>
            <h2 id="web-sec">Developing Web Applications</h2>
            <p>Use the Spring MVC starter to build REST APIs.</p>
            <div class="listingblock">
                <div class="content">
                    <pre class="highlight"><code class="language-java">@RestController
public class MyController {
    @GetMapping("/hello")
    public String hello() {
        return "Hello World";
    }
}</code></pre>
                </div>
            </div>
        </article>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """

    parser = DocumentParser()
    sections = parser.parse_html(html_sample)

    assert len(sections) >= 2
    # Verify script, nav, footer removed
    full_text = " ".join(s.content for s in sections)
    assert "malicious" not in full_text
    assert "Copyright 2026" not in full_text
    assert "Header content" not in full_text

    # Verify java code block preserved
    assert "```java" in full_text
    assert "@RestController" in full_text
    assert 'return "Hello World";' in full_text


def test_parse_markdown_splits_by_headers() -> None:
    """Test markdown parser splitting by headers and preserving code inside."""
    md_sample = """# Spring Boot Core
Spring Boot core features include auto-configuration.

## Auto-configuration
Auto-configuration attempts to automatically configure beans.

```java
@Configuration
@EnableAutoConfiguration
public class Application {}
```

## Externalized Configuration
Spring Boot lets you externalize your configuration.
"""
    parser = DocumentParser()
    sections = parser.parse_markdown(md_sample)

    assert len(sections) == 3
    assert sections[0].title == "Spring Boot Core"
    assert sections[1].title == "Auto-configuration"
    assert "```java" in sections[1].content
    assert sections[2].title == "Externalized Configuration"
