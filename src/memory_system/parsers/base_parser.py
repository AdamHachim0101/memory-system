"""
Base Parser for Source Documents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ParsedContent:
    """Result of parsing a document."""
    text: str
    structure: List[Dict[str, Any]]  # Hierarchical structure
    metadata: Dict[str, Any]
    page_count: Optional[int] = None
    language: Optional[str] = None


@dataclass
class Section:
    """A section within a document."""
    title: Optional[str]
    level: int
    content: str
    path: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    children: List['Section'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    supported_types = []

    @classmethod
    def can_parse(cls, mime_type: str, filename: str) -> bool:
        """Check if this parser can handle the given file type."""
        for ext in cls.supported_types:
            if filename.lower().endswith(ext):
                return True
        return mime_type in cls.supported_types

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """
        Parse document content.

        Args:
            content: Raw bytes of the document
            filename: Original filename

        Returns:
            ParsedContent with text, structure, and metadata
        """
        pass

    def _count_tokens(self, text: str) -> int:
        """Rough token estimation (chars / 4 for English)."""
        return len(text) // 4

    def _split_into_chunks(
        self,
        content: str,
        max_tokens: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split content into chunks of approximately max_tokens.

        Args:
            content: Text to split
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks
        """
        chunk_size = max_tokens * 4  # chars per chunk (rough estimate)
        overlap_chars = overlap * 4

        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            chunks.append(chunk)
            start = end - overlap_chars

        return chunks