"""
Platform Context Agent — enriches the pipeline state with IDRE platform knowledge.

Sits between schema_mapper and sql_writer in the pipeline. Gathers relevant
platform context (business rules, calculations, report logic, code intelligence)
based on the user's query and injects it into the state for the SQL writer.

This agent makes NO LLM calls — it's a pure knowledge lookup.
"""
from state.context import GraphState
from knowledge.knowledge_base import (
    get_platform_context_for_query,
    search_by_concepts,
    load_platform_rules,
)


def _extract_query_concepts(query: str) -> list[str]:
    """Extract domain concepts from the query for code intelligence lookup."""
    query_lower = query.lower()
    concept_keywords = {
        "payment": ["payment", "pay", "fee", "amount", "fund", "refund", "disbursement"],
        "case": ["case", "dispute", "filing", "claim"],
        "arbitration": ["arbitration", "decision", "determination", "award", "arbitrator"],
        "eligibility": ["eligibility", "eligible", "qualify", "review", "rfi"],
        "organization": ["organization", "org", "provider", "health plan", "insurer"],
        "invoice": ["invoice", "billing", "cms"],
        "banking": ["bank", "ach", "nacha", "routing"],
        "report": ["report", "analytics", "dashboard", "export", "summary"],
    }
    matched = []
    for concept, keywords in concept_keywords.items():
        if any(kw in query_lower for kw in keywords):
            matched.append(concept)
    return matched


def _get_code_intelligence_context(concepts: list[str], max_files: int = 5) -> str:
    """Find relevant code files from IDRE codebase summaries."""
    if not concepts:
        return ""
    files = search_by_concepts(concepts, top_k=max_files)
    if not files:
        return ""

    lines = ["=== Relevant IDRE Platform Code Context ==="]
    for f in files:
        purpose = f.get("purpose", "")
        path = f.get("path", "")
        exports = f.get("key_exports", [])
        api_routes = f.get("api_routes", [])
        if purpose:
            line = f"  {path}: {purpose}"
            if api_routes:
                line += f" (routes: {', '.join(api_routes[:3])})"
            if exports:
                line += f" (exports: {', '.join(exports[:3])})"
            lines.append(line)
    return "\n".join(lines)


def platform_context_node(state: GraphState) -> GraphState:
    """
    Gather platform-specific context relevant to the user's query.
    This context helps the SQL writer understand business rules,
    calculations, and report logic from the IDRE platform.
    """
    query = state.get("resolved_query") or state["user_query"]

    # 1. Get business rules context (statuses, pricing, calculations, etc.)
    rules_context = get_platform_context_for_query(query)

    # 2. Get code intelligence context (relevant source files)
    concepts = _extract_query_concepts(query)
    code_context = _get_code_intelligence_context(concepts)

    # Combine contexts
    context_parts = []
    if rules_context:
        context_parts.append(rules_context)
    if code_context:
        context_parts.append(code_context)

    platform_context = "\n\n".join(context_parts) if context_parts else ""

    # Build trace entry
    detail = []
    if rules_context:
        rule_sections = [line for line in rules_context.split("\n") if line.startswith("===")]
        detail.append(f"Business rules: {len(rule_sections)} section(s) matched")
    if code_context:
        file_count = code_context.count("  ")
        detail.append(f"Code intelligence: {file_count} relevant file(s) found")
    if concepts:
        detail.append(f"Query concepts: {', '.join(concepts)}")

    summary = "Platform context assembled"
    if not platform_context:
        summary = "No specific platform context matched — using schema only"

    trace_entry = {
        "agent": "Platform Context",
        "status": "ok" if platform_context else "warn",
        "summary": summary,
        "detail": detail,
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {
        **state,
        "platform_context": platform_context,
        "agent_trace": trace,
    }
