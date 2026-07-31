"""PMC 语义分析 Markdown 文档的固定输出契约。"""

from __future__ import annotations

from .model import PmcModel


MARKDOWN_DOCUMENT_SCHEMA_VERSION = "mnn-pmc-markdown-document/v1"


class MarkdownSection(PmcModel):
    title: str
    content: str


class MarkdownDocument(PmcModel):
    schema_version: str = MARKDOWN_DOCUMENT_SCHEMA_VERSION
    title: str
    sections: tuple[MarkdownSection, ...]
    warnings: tuple[str, ...]
    markdown: str
