"""
JSON Parser for JSON documents
"""

import json
from typing import Any
from .base_parser import BaseParser, ParsedContent


class JSONParser(BaseParser):
    """Parser for JSON files (.json)."""

    supported_types = ['.json']

    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """Parse JSON content."""
        text = content.decode('utf-8', errors='replace')

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

        text_content = self._flatten_to_text(data)

        structure = [{
            'type': 'document',
            'title': filename,
            'level': 0,
            'content': text_content[:500]
        }]

        metadata = {
            'parser': 'json',
            'encoding': 'utf-8',
            'char_count': len(text_content),
            'keys': list(data.keys()) if isinstance(data, dict) else []
        }

        return ParsedContent(
            text=text_content,
            structure=structure,
            metadata=metadata,
            page_count=None,
            language=None
        )

    def _flatten_to_text(self, obj: Any, depth: int = 0) -> str:
        """Flatten JSON object to readable text."""
        if depth > 10:
            return str(obj)

        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(f"{k}: {self._flatten_to_text(v, depth + 1)}")
            return ', '.join(parts)
        elif isinstance(obj, list):
            return ', '.join([self._flatten_to_text(item, depth + 1) for item in obj])
        else:
            return str(obj)
