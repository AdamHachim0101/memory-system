from memory_system.models import RetrievalProfile


CONTEXT_BUDGETS = {
    RetrievalProfile.LIGHT: {
        "max_tokens": 2000,
        "max_semantic_memories": 3,
        "max_episodic_snippets": 2,
        "max_digests": 0,
        "max_graph_context": 0,
    },
    RetrievalProfile.TASK: {
        "max_tokens": 4000,
        "max_semantic_memories": 6,
        "max_episodic_snippets": 4,
        "max_digests": 1,
        "max_graph_context": 2,
    },
    RetrievalProfile.DEEP: {
        "max_tokens": 8000,
        "max_semantic_memories": 8,
        "max_episodic_snippets": 6,
        "max_digests": 2,
        "max_graph_context": 5,
    },
    RetrievalProfile.AUDIT: {
        "max_tokens": 12000,
        "max_semantic_memories": 15,
        "max_episodic_snippets": 10,
        "max_digests": 3,
        "max_graph_context": 10,
    },
}


def pack_context(
    working_memory: dict | None,
    semantic_memories: list[dict],
    episodic_events: list[dict],
    digests: list[dict],
    graph_context: list[dict],
    profile: RetrievalProfile,
) -> str:
    budget = CONTEXT_BUDGETS[profile]
    sections = []

    if working_memory:
        sections.append(format_working_memory(working_memory))

    if semantic_memories:
        semantic_text = format_semantic_memories(
            semantic_memories[: budget["max_semantic_memories"]]
        )
        if semantic_text:
            sections.append(semantic_text)

    if episodic_events:
        episodic_text = format_episodic_events(
            episodic_events[: budget["max_episodic_snippets"]]
        )
        if episodic_text:
            sections.append(episodic_text)

    if digests:
        digest_text = format_digests(digests[: budget["max_digests"]])
        if digest_text:
            sections.append(digest_text)

    if graph_context and profile != RetrievalProfile.LIGHT:
        graph_text = format_graph_context(
            graph_context[: budget["max_graph_context"]]
        )
        if graph_text:
            sections.append(graph_text)

    return "\n\n---\n\n".join(sections)


def format_working_memory(working_memory: dict) -> str:
    lines = ["## Working Memory (Current Task State)"]
    if working_memory.get("objective"):
        lines.append(f"- **Current Objective**: {working_memory['objective']}")
    if working_memory.get("active_tasks"):
        lines.append("- **Active Tasks**:")
        for task in working_memory["active_tasks"]:
            lines.append(f"  - {task}")
    if working_memory.get("constraints"):
        lines.append("- **Constraints**:")
        for constraint in working_memory["constraints"]:
            lines.append(f"  - {constraint}")
    if working_memory.get("open_questions"):
        lines.append("- **Open Questions**:")
        for q in working_memory["open_questions"]:
            lines.append(f"  - {q}")
    if working_memory.get("summary"):
        lines.append(f"- **Summary**: {working_memory['summary']}")
    return "\n".join(lines)


def format_semantic_memories(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = ["## Relevant Memories"]
    for memory in memories:
        mem_type = memory.get("memory_type", "fact")
        canonical = memory.get("canonical_text", "")
        confidence = memory.get("confidence", 0.5)
        lines.append(f"- [{mem_type}] {canonical} (confidence: {confidence:.0%})")
    return "\n".join(lines)


def format_episodic_events(events: list[dict]) -> str:
    if not events:
        return ""
    lines = ["## Recent Conversation History"]
    for event in events:
        role = event.get("role", "unknown")
        content = event.get("content", "")[:200]
        turn_id = event.get("turn_id", "?")
        lines.append(f"- **Turn {turn_id}** ({role}): {content}")
    return "\n".join(lines)


def format_digests(digests: list[dict]) -> str:
    if not digests:
        return ""
    lines = ["## Session/Context Digests"]
    for digest in digests:
        title = digest.get("title", "Digest")
        summary = digest.get("summary", "")[:300]
        period = digest.get("period_type", "session")
        lines.append(f"### [{period.upper()}] {title}")
        lines.append(summary)
        if digest.get("decisions"):
            lines.append("**Decisions**:")
            for d in digest["decisions"][:3]:
                lines.append(f"  - {d}")
    return "\n".join(lines)


def format_graph_context(graph_data: list[dict]) -> str:
    if not graph_data:
        return ""
    lines = ["## Entity Relationships"]
    seen = set()
    for item in graph_data:
        e1 = item.get("e1", {})
        e2 = item.get("e2", {})
        if e1 and e2:
            key = f"{e1.get('type')}:{e1.get('value')}->{e2.get('type')}:{e2.get('value')}"
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"- **{e1.get('value', '?')}** ({e1.get('type', 'entity')}) "
                    f"-> **{e2.get('value', '?')}** ({e2.get('type', 'entity')})"
                )
    return "\n".join(lines)
