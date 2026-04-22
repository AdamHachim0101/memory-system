"""
PDF Parser for PDF documents
"""

from typing import List, Optional
from .base_parser import BaseParser, ParsedContent


class PDFParser(BaseParser):
    """Parser for PDF files (.pdf)."""

    supported_types = ['.pdf']

    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """Parse PDF content using PyPDF2 or pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                raise ImportError(
                    "PDF parsing requires 'pypdf' or 'PyPDF2' library. "
                    "Install with: pip install pypdf"
                )

        from io import BytesIO

        reader = PdfReader(BytesIO(content))

        pages = []
        all_text = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append({
                'page_num': i + 1,
                'text': text
            })
            all_text.append(text)

        full_text = '\n\n'.join(all_text)

        structure = []
        for page in pages:
            structure.append({
                'type': 'page',
                'page_num': page['page_num'],
                'level': 0,
                'content': page['text'][:200]
            })

        metadata = {
            'parser': 'pdf',
            'page_count': len(pages),
            'char_count': len(full_text)
        }

        return ParsedContent(
            text=full_text,
            structure=structure,
            metadata=metadata,
            page_count=len(pages),
            language=None
        )


def pdf_parser_availability() -> bool:
    """Check if PDF parsing is available."""
    try:
        from pypdf import PdfReader
        return True
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            return True
        except ImportError:
            return False
