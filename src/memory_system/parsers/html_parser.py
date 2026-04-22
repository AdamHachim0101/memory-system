"""
HTML Parser for web pages and HTML documents
"""

import re
from typing import List
from .base_parser import BaseParser, ParsedContent


class HTMLParser(BaseParser):
    """Parser for HTML files (.html, .htm)."""

    supported_types = ['.html', '.htm']

    SCRIPT_STYLE_PATTERN = re.compile(
        r'<(script|style)[^>]*>.*?</\1>',
        re.DOTALL | re.IGNORECASE
    )
    TAG_PATTERN = re.compile(r'<[^>]+>')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """Parse HTML content and extract text."""
        html = content.decode('utf-8', errors='replace')

        text = self._extract_text(html)

        structure = [{
            'type': 'document',
            'title': filename,
            'level': 0,
            'content': text[:500]
        }]

        headings = self._extract_headings(html)
        for h in headings:
            structure.append({
                'type': 'section',
                'title': h['text'],
                'level': h['level'],
                'path': f"/{h['text']}",
                'content': ''
            })

        metadata = {
            'parser': 'html',
            'encoding': 'utf-8',
            'char_count': len(text),
            'heading_count': len(headings)
        }

        return ParsedContent(
            text=text,
            structure=structure,
            metadata=metadata,
            page_count=None,
            language=None
        )

    def _extract_text(self, html: str) -> str:
        """Remove scripts, styles, and tags to extract readable text."""
        text = self.SCRIPT_STYLE_PATTERN.sub('', html)
        text = self.TAG_PATTERN.sub(' ', text)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()

    def _extract_headings(self, html: str) -> List[dict]:
        """Extract headings from HTML."""
        headings = []
        pattern = re.compile(r'<h([1-6])[^>]*>([^<]+)</h\1>', re.IGNORECASE)

        for match in pattern.finditer(html):
            level = int(match.group(1))
            text = match.group(2).strip()
            headings.append({'level': level, 'text': text})

        return headings
