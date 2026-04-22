"""
Prompt Composer for Source Workspace Engine
Builds prompts with separated contexts for hybrid queries
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AgentMemoryContext:
    """Context from the agent's 5-memory system."""
    working_memory: str = ""
    semantic_memories: str = ""
    episodic_snippets: str = ""
    recent_facts: str = ""
    timeline: str = ""


@dataclass
class SourceEvidenceContext:
    """Context from the Source Workspace."""
    documents: List[Dict[str, Any]] = None
    chunks: List[Dict[str, Any]] = None
    citations: Dict[str, str] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.chunks is None:
            self.chunks = []
        if self.citations is None:
            self.citations = {}


@dataclass
class HybridPrompt:
    """Composed hybrid prompt."""
    full_prompt: str
    agent_memory_block: str
    source_evidence_block: str
    instructions_block: str
    token_count: int


class PromptComposer:
    """
    Composes prompts with clearly separated blocks:
    - Agent Memory Context
    - Source Evidence Context
    - Instructions

    This separation ensures:
    1. Agent memory is used for conversational continuity
    2. Source evidence is used for factual claims with citations
    3. No contamination between memory and external evidence
    """

    @staticmethod
    def compose_hybrid(
        agent_memory: AgentMemoryContext,
        source_evidence: SourceEvidenceContext,
        user_query: str,
        include_citations: bool = True,
        max_source_tokens: int = 2000
    ) -> HybridPrompt:
        """
        Compose a hybrid prompt with separated contexts.

        Args:
            agent_memory: Context from agent's 5-memory system
            source_evidence: Context from source workspace
            user_query: The user's query
            include_citations: Whether to include source citations
            max_source_tokens: Maximum tokens for source context

        Returns:
            HybridPrompt with all blocks
        """
        agent_block = PromptComposer._build_agent_memory_block(agent_memory)
        source_block = PromptComposer._build_source_evidence_block(
            source_evidence,
            max_source_tokens,
            include_citations
        )
        instructions_block = PromptComposer._build_instructions_block(
            include_citations
        )

        full_prompt = f"""{agent_block}

---

{source_block}

---

{instructions_block}

---

**User Query:** {user_query}

**Your Response:**"""

        return HybridPrompt(
            full_prompt=full_prompt,
            agent_memory_block=agent_block,
            source_evidence_block=source_block,
            instructions_block=instructions_block,
            token_count=len(full_prompt) // 4
        )

    @staticmethod
    def _build_agent_memory_block(memory: AgentMemoryContext) -> str:
        """Build the agent memory context block."""
        sections = ["**[AGENT MEMORY CONTEXT]**"]

        if memory.working_memory:
            sections.append(f"\n**Working Memory:**\n{memory.working_memory}")

        if memory.semantic_memories:
            sections.append(f"\n**Semantic Memories:**\n{memory.semantic_memories}")

        if memory.episodic_snippets:
            sections.append(f"\n**Recent Conversation:**\n{memory.episodic_snippets}")

        if memory.recent_facts:
            sections.append(f"\n**Facts from Conversation:**\n{memory.recent_facts}")

        if memory.timeline:
            sections.append(f"\n**Timeline:**\n{memory.timeline}")

        if len(sections) == 1:
            sections.append("\n_No relevant agent memory available_")

        return "\n".join(sections)

    @staticmethod
    def _build_source_evidence_block(
        evidence: SourceEvidenceContext,
        max_tokens: int,
        include_citations: bool
    ) -> str:
        """Build the source evidence context block."""
        sections = ["**[SOURCE EVIDENCE CONTEXT]**"]

        if not evidence.documents and not evidence.chunks:
            sections.append("\n_No source documents available_")
            return "\n".join(sections)

        doc_titles = {}

        for doc in evidence.documents:
            doc_id = doc.get('source_id', '')
            title = doc.get('title', 'Unknown Document')
            doc_titles[doc_id] = title
            sections.append(f"\n**Document:** {title}")

            if doc.get('summary'):
                sections.append(f"Summary: {doc['summary'][:200]}...")

        if evidence.chunks:
            sections.append("\n**Relevant Content:**")

            total_chars = 0
            for i, chunk in enumerate(evidence.chunks, 1):
                source_id = chunk.get('source_id', '')
                source_title = doc_titles.get(source_id, 'Unknown Source')
                content = chunk.get('content', '')
                page = chunk.get('page')

                page_str = f" (page {page})" if page else ""

                truncated = content[:800] + "..." if len(content) > 800 else content

                sections.append(
                    f"\n[{i}] Source: {source_title}{page_str}\n"
                    f"{truncated}"
                )

                total_chars += len(truncated)

                if total_chars > max_tokens * 4:
                    sections.append("\n_(truncated - too many chunks)_")
                    break

        if include_citations and evidence.citations:
            sections.append("\n**Citations:**")
            for chunk_id, citation in evidence.citations.items():
                sections.append(f"- {citation}")

        return "\n".join(sections)

    @staticmethod
    def _build_instructions_block(include_citations: bool) -> str:
        """Build the instructions block."""
        instructions = [
            "**[INSTRUCTIONS]**",
            "1. Use agent memory context for conversational continuity",
            "2. Use source evidence for factual claims and specific information",
            "3. When citing sources, reference the document title and page if available",
            "4. Do not confuse agent memory with external source evidence",
            "5. If information comes from sources, attribute it accordingly"
        ]

        if not include_citations:
            instructions[3] = "3. Reference sources by name when using their content"

        return "\n".join(instructions)

    @staticmethod
    def compose_source_only(
        source_evidence: SourceEvidenceContext,
        user_query: str,
        include_citations: bool = True
    ) -> HybridPrompt:
        """
        Compose a prompt for source-only queries (no agent memory).
        """
        source_block = PromptComposer._build_source_evidence_block(
            source_evidence,
            3000,
            include_citations
        )

        instructions = [
            "**[INSTRUCTIONS]**",
            "1. Answer based solely on the provided source evidence",
            "2. If the evidence doesn't contain enough information, say so",
            "3. When using content from sources, cite them appropriately"
        ]

        full_prompt = f"""{source_block}

---

{"\n".join(instructions)}

---

**User Query:** {user_query}

**Your Response:**"""

        return HybridPrompt(
            full_prompt=full_prompt,
            agent_memory_block="",
            source_evidence_block=source_block,
            instructions_block="\n".join(instructions),
            token_count=len(full_prompt) // 4
        )

    @staticmethod
    def compose_memory_only(
        agent_memory: AgentMemoryContext,
        user_query: str
    ) -> HybridPrompt:
        """
        Compose a prompt for memory-only queries (no source evidence).
        """
        agent_block = PromptComposer._build_agent_memory_block(agent_memory)

        instructions = [
            "**[INSTRUCTIONS]**",
            "1. Answer based on your memory and conversation history",
            "2. If you don't have relevant memory, say so honestly"
        ]

        full_prompt = f"""{agent_block}

---

{"\n".join(instructions)}

---

**User Query:** {user_query}

**Your Response:**"""

        return HybridPrompt(
            full_prompt=full_prompt,
            agent_memory_block=agent_block,
            source_evidence_block="",
            instructions_block="\n".join(instructions),
            token_count=len(full_prompt) // 4
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough: chars / 4)."""
        return len(text) // 4
