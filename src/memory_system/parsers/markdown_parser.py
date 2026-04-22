"""
Markdown Parser for markdown documents
"""

import re
from typing import List
from .base_parser import BaseParser, ParsedContent, Section


class MarkdownParser(BaseParser):
    """Parser for Markdown files (.md, .mdx)."""

    supported_types = ['.md', '.mdx', '.markdown']

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    async def parse(self, content: bytes, filename: str) -> ParsedContent:
        """Parse markdown content and extract structure."""
        text = content.decode('utf-8', errors='replace')

        sections = self._extract_sections(text)

        structure = []
        for section in sections:
            structure.append({
                'type': 'section',
                'title': section.title,
                'level': section.level,
                'path': section.path,
                'content': section.content,
                'page_start': section.page_start,
                'page_end': section.page_end
            })

        metadata = {
            'parser': 'markdown',
            'encoding': 'utf-8',
            'char_count': len(text),
            'section_count': len(sections),
            'heading_count': len(self.HEADING_PATTERN.findall(text))
        }

        return ParsedContent(
            text=text,
            structure=structure,
            metadata=metadata,
            page_count=None,
            language=None
        )

    def _extract_sections(self, text: str) -> List[Section]:
        """Extract hierarchical sections from markdown."""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match:
                if current_section is not None:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                path = self._build_path(sections, title, level)

                current_section = Section(
                    title=title,
                    level=level,
                    content='',
                    path=path,
                    children=[]
                )
                current_content = []
            else:
                current_content.append(line)

        if current_section is not None:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        if not sections:
            sections.append(Section(
                title='Document',
                level=0,
                content=text.strip(),
                path='/Document'
            ))

        return sections

    def _build_path(self, sections: List[Section], title: str, level: int) -> str:
        """Build section path based on hierarchy."""
        path_parts = []

        for s in sections:
            if s.level < level:
                path_parts.append(s.title or 'Untitled')

        path_parts.append(title)

        return '/' + '/'.join(path_parts)
