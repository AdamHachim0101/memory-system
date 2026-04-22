"""
Text Parser for plain text documents
"""

from typing import List
from .base_parser import BaseParser, ParsedContent


class TextParser(BaseParser):
    """Parser for plain text files (.txt, .log, etc.)."""

    supported_types = ['.txt', '.log', '.text']

    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """Parse plain text content."""
        text = content.decode('utf-8', errors='replace')

        structure = [{
            'type': 'document',
            'title': filename,
            'level': 0,
            'content': text
        }]

        metadata = {
            'parser': 'text',
            'encoding': 'utf-8',
            'char_count': len(text),
            'line_count': text.count('\n') + 1
        }

        return ParsedContent(
            text=text,
            structure=structure,
            metadata=metadata,
            page_count=None,
            language=None
        )
