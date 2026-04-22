"""
Citation Engine for Source Workspace Engine
Generates citations and manages source references
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Citation:
    """A citation for a source chunk."""
    citation_id: str
    chunk_id: str
    source_id: str
    source_title: str
    page: Optional[int]
    locator: str
    quote_text: str
    created_at: datetime


@dataclass
class GroundedEvidence:
    """Evidence from sources for grounding responses."""
    content: str
    citation: Citation
    relevance_score: float


class CitationEngine:
    """
    Engine for creating and managing citations.

    Citation format support:
    - APA style
    - Custom formats
    - Direct source reference
    """

    @staticmethod
    def create_citation(
        chunk_id: str,
        source_id: str,
        source_title: str,
        page: Optional[int],
        locator: str,
        quote_text: str
    ) -> Citation:
        """Create a new citation."""
        import uuid
        return Citation(
            citation_id=str(uuid.uuid4()),
            chunk_id=chunk_id,
            source_id=source_id,
            source_title=source_title,
            page=page,
            locator=locator,
            quote_text=quote_text,
            created_at=datetime.utcnow()
        )

    @staticmethod
    def format_apa(citation: Citation) -> str:
        """
        Format citation in APA style.

        Example:
        Smith, J. (2024). Document Title. p. 5.
        """
        page_str = f", p. {citation.page}" if citation.page else ""
        return f"{citation.source_title}{page_str}"

    @staticmethod
    def format_mla(citation: Citation) -> str:
        """
        Format citation in MLA style.

        Example:
        Smith, John. "Document Title." p. 5.
        """
        page_str = f", p. {citation.page}" if citation.page else ""
        return f'"{citation.source_title}"{page_str}'

    @staticmethod
    def format_direct(citation: Citation) -> str:
        """
        Format citation as direct reference.

        Example:
        [Source: Document Title, Page 5]
        """
        page_str = f", Page {citation.page}" if citation.page else ""
        return f"[Source: {citation.source_title}{page_str}]"

    @staticmethod
    def format_custom(citation: Citation, template: str) -> str:
        """
        Format citation using custom template.

        Variables: {source}, {page}, {locator}, {quote}
        """
        return template.format(
            source=citation.source_title,
            page=citation.page or "",
            locator=citation.locator,
            quote=citation.quote_text[:100]
        )

    @staticmethod
    def build_evidence(
        chunks: List[Dict[str, Any]],
        source_titles: Dict[str, str],
        relevance_scores: List[float]
    ) -> List[GroundedEvidence]:
        """
        Build grounded evidence from retrieved chunks.

        Args:
            chunks: List of retrieved chunks
            source_titles: Mapping of source_id to title
            relevance_scores: Relevance scores for each chunk

        Returns:
            List of GroundedEvidence with citations
        """
        evidence = []

        for i, chunk in enumerate(chunks):
            source_id = chunk.get('source_id', '')
            chunk_id = chunk.get('chunk_id', '')

            citation = CitationEngine.create_citation(
                chunk_id=chunk_id,
                source_id=source_id,
                source_title=source_titles.get(source_id, 'Unknown Source'),
                page=chunk.get('page'),
                locator=f"chunk_{i}",
                quote_text=chunk.get('content', '')[:300]
            )

            score = relevance_scores[i] if i < len(relevance_scores) else 0.5

            evidence.append(GroundedEvidence(
                content=chunk.get('content', ''),
                citation=citation,
                relevance_score=score
            ))

        evidence.sort(key=lambda x: x.relevance_score, reverse=True)

        return evidence

    @staticmethod
    def format_response_with_citations(
        response: str,
        evidence: List[GroundedEvidence],
        format_style: str = "direct"
    ) -> str:
        """
        Format a response with inline citations.

        Args:
            response: The generated response text
            evidence: List of grounded evidence
            format_style: Style of citation ('apa', 'mla', 'direct')

        Returns:
            Response with citations appended
        """
        if not evidence:
            return response

        citations = []

        for i, ev in enumerate(evidence, 1):
            if format_style == "apa":
                cit_str = CitationEngine.format_apa(ev.citation)
            elif format_style == "mla":
                cit_str = CitationEngine.format_mla(ev.citation)
            else:
                cit_str = CitationEngine.format_direct(ev.citation)

            citations.append(f"[{i}] {cit_str}")

        citations_text = "\n\n**Sources:**\n" + "\n".join(citations)

        return response + citations_text

    @staticmethod
    def build_citation_context(
        chunks: List[Dict[str, Any]],
        source_titles: Dict[str, str]
    ) -> str:
        """
        Build a context string with citations for use in prompts.

        Args:
            chunks: Retrieved chunks
            source_titles: Source ID to title mapping

        Returns:
            Formatted context with citations
        """
        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source_id = chunk.get('source_id', '')
            source_title = source_titles.get(source_id, 'Unknown')
            page = chunk.get('page', '')
            content = chunk.get('content', '')

            page_str = f" (page {page})" if page else ""

            context_parts.append(
                f"[{i}] Source: {source_title}{page_str}\n"
                f"Content: {content[:500]}..."
            )

        return "\n\n".join(context_parts)