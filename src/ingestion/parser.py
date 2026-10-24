"""Semantic documentation parser and markup cleaner."""

import re

from bs4 import BeautifulSoup, Tag


class ParsedSection:
    """Represents an extracted semantic section from documentation."""

    def __init__(
        self,
        title: str,
        content: str,
        level: int = 2,
        anchor: str | None = None,
        parent_title: str | None = None,
    ) -> None:
        self.title = title
        self.content = content
        self.level = level
        self.anchor = anchor
        self.parent_title = parent_title

    def __repr__(self) -> str:
        return f"<ParsedSection '{self.title}' (level={self.level})>"


class DocumentParser:
    """Parser for Spring Boot documentation in HTML or Markdown format."""

    UNWANTED_TAGS: list[str] = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "svg",
    ]

    UNWANTED_CLASSES: list[str] = [
        "toc",
        "navigation",
        "sidebar",
        "breadcrumbs",
        "page-footer",
        "feedback-prompt",
    ]

    def parse_html(
        self, html_content: str, default_title: str = "Documentation"
    ) -> list[ParsedSection]:
        """Parse HTML documentation into clean, semantic markdown sections."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove scripts, styles, headers, footers and navigation
        for tag_name in self.UNWANTED_TAGS:
            for elem in soup.find_all(tag_name):
                elem.decompose()

        for class_name in self.UNWANTED_CLASSES:
            for elem in soup.find_all(class_=re.compile(class_name, re.I)):
                elem.decompose()

        main_content = soup.find("article") or soup.find("main") or soup.find("body") or soup
        sections: list[ParsedSection] = []

        current_title = default_title
        current_level = 1
        current_anchor = None
        current_parent: str | None = None
        buffer_lines: list[str] = []

        # Find first h1 if available
        h1 = main_content.find("h1")
        if h1:
            current_title = h1.get_text(strip=True)

        for element in main_content.descendants:
            if not isinstance(element, Tag):
                continue

            if element.name in ["h1", "h2", "h3", "h4"]:
                # Flush previous section
                text_block = "\n".join(buffer_lines).strip()
                if text_block or len(sections) == 0:
                    sections.append(
                        ParsedSection(
                            title=current_title,
                            content=text_block,
                            level=current_level,
                            anchor=current_anchor,
                            parent_title=current_parent,
                        )
                    )
                    buffer_lines = []

                header_text = element.get_text(strip=True)
                level = int(element.name[1])
                if level <= 2:
                    current_parent = header_text
                current_title = header_text
                current_level = level
                current_anchor = element.get("id")

            elif element.name == "pre":
                # Ensure we don't process nested tags inside pre separately
                code_text = self._format_code_block(element)
                if code_text:
                    buffer_lines.append(code_text)

            elif element.name in ["p", "li", "table"]:
                # Ignore elements that are nested inside pre or another p/li/table
                if element.find_parent("pre") or element.find_parents(["p", "li", "table"]):
                    continue
                block_text = self._element_to_text(element)
                if block_text:
                    buffer_lines.append(block_text)

        # Flush final section
        final_block = "\n".join(buffer_lines).strip()
        if final_block:
            sections.append(
                ParsedSection(
                    title=current_title,
                    content=final_block,
                    level=current_level,
                    anchor=current_anchor,
                    parent_title=current_parent,
                )
            )

        # Filter out empty sections
        return [s for s in sections if s.content.strip()]

    def parse_markdown(
        self, markdown_content: str, default_title: str = "Documentation"
    ) -> list[ParsedSection]:
        """Parse Markdown documentation split by headers (# and ##)."""
        lines = markdown_content.splitlines()
        sections: list[ParsedSection] = []

        current_title = default_title
        current_level = 1
        current_parent: str | None = None
        buffer_lines: list[str] = []
        in_code_block = False

        header_regex = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                buffer_lines.append(line)
                continue

            match = header_regex.match(line)
            if match and not in_code_block:
                text_block = "\n".join(buffer_lines).strip()
                if text_block:
                    sections.append(
                        ParsedSection(
                            title=current_title,
                            content=text_block,
                            level=current_level,
                            parent_title=current_parent,
                        )
                    )
                    buffer_lines = []

                level = len(match.group(1))
                title = match.group(2).strip()
                if level <= 2:
                    current_parent = title
                current_title = title
                current_level = level
            else:
                buffer_lines.append(line)

        final_block = "\n".join(buffer_lines).strip()
        if final_block:
            sections.append(
                ParsedSection(
                    title=current_title,
                    content=final_block,
                    level=current_level,
                    parent_title=current_parent,
                )
            )

        return [s for s in sections if s.content.strip()]

    def _format_code_block(self, pre: Tag) -> str:
        """Format a <pre> tag into a markdown code block with language identifier."""
        code_elem = pre.find("code")
        lang = ""

        targets = [code_elem, pre] if code_elem else [pre]
        for target in targets:
            if not target:
                continue
            classes = target.get("class", [])
            for cls in classes:
                if cls.startswith("language-") or cls.startswith("lang-"):
                    lang = cls.split("-", 1)[1]
                    break
            if lang:
                break

        code_text = code_elem.get_text() if code_elem else pre.get_text()
        code_text = code_text.strip()

        if not lang:
            # Heuristic detection for common Spring Boot patterns
            if (
                "import org.springframework" in code_text
                or "@SpringBoot" in code_text
                or "public class" in code_text
            ):
                lang = "java"
            elif "spring:" in code_text or "server:" in code_text:
                lang = "yaml"
            elif "spring." in code_text or "server." in code_text:
                lang = "properties"
            elif "<project" in code_text or "<dependencies" in code_text:
                lang = "xml"

        return f"\n```{lang}\n{code_text}\n```\n"

    def _element_to_text(self, element: Tag) -> str:
        """Convert a BeautifulSoup tag to cleaned text."""
        raw_text = element.get_text()
        return re.sub(r"[ \t]+", " ", raw_text).strip()
