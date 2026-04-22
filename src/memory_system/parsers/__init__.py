"""
Parser Factory for Source Documents
"""

from typing import Optional, List
from .base_parser import BaseParser, ParsedContent
from .text_parser import TextParser
from .markdown_parser import MarkdownParser
from .html_parser import HTMLParser
from .json_parser import JSONParser
from .pdf_parser import PDFParser


class ParserFactory:
    """Factory for creating appropriate parsers based on file type."""

    _parsers: List[BaseParser] = []
    _initialized = False

    @classmethod
    def _initialize(cls):
        """Lazy initialization of parsers."""
        if cls._initialized:
            return

        from .text_parser import TextParser
        from .markdown_parser import MarkdownParser
        from .html_parser import HTMLParser
        from .json_parser import JSONParser

        cls._parsers = [
            TextParser(),
            MarkdownParser(),
            HTMLParser(),
            JSONParser(),
        ]

        try:
            from .pdf_parser import PDFParser
            cls._parsers.append(PDFParser())
        except ImportError:
            pass

        cls._initialized = True

    @classmethod
    def get_parser(cls, filename: str, mime_type: Optional[str] = None) -> Optional[BaseParser]:
        """
        Get the appropriate parser for a file.

        Args:
            filename: Name of the file
            mime_type: MIME type of the file

        Returns:
            Parser instance or None if no parser found
        """
        cls._initialize()

        for parser in cls._parsers:
            if parser.can_parse(mime_type or '', filename):
                return parser

        return None

    @classmethod
    async def parse(
        cls,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None
    ) -> ParsedContent:
        """
        Parse content using the appropriate parser.

        Args:
            content: Raw bytes of the document
            filename: Original filename
            mime_type: MIME type if known

        Returns:
            ParsedContent with text, structure, and metadata

        Raises:
            ValueError: If no parser found for file type
        """
        parser = cls.get_parser(filename, mime_type)

        if parser is None:
            raise ValueError(f"No parser available for file: {filename}")

        return await parser.parse(content, filename)

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Get list of all supported file extensions."""
        cls._initialize()

        extensions = []
        for parser in cls._parsers:
            extensions.extend(parser.supported_types)

        return list(set(extensions))

    @classmethod
    def available_parsers(cls) -> List[str]:
        """Get list of available parser names."""
        cls._initialize()

        return [p.__class__.__name__.replace('Parser', '') for p in cls._parsers]
