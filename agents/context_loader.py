"""
Context Loader Agent  (Story 2.2 — Within-Session Memory & Pronoun Resolution)

Sits at the front of the pipeline.  Given the raw user_query and the
conversation_history from the current session, it rewrites the query into a
fully self-contained question that downstream agents can process without any
prior context.

Fast path: if history is empty, or the query contains no reference words,
resolved_query = user_query (no LLM call).
"""
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings
from state.context import GraphState
from utils.glossary_matcher import find_matches
from utils.permissions import get_permitted_tables, get_role_display

# Words that signal the query may refer to a prior turn
_REFERENCE_PATTERN = re.compile(
    r"\b(it|its|they|them|their|those|these|that|this|"
    r"same|similar|such|the previous|the last|the above|"
    r"those cases|that query|same filter|same status|"
    r"what about|how about)\b"
    r"|^\s*and\b",          # query starting with "and ..." is always a continuation
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a query resolver for a data analytics chatbot about dispute resolution cases.

Given a short conversation history and a new user message, rewrite the message as a \
fully self-contained question that can be understood with no prior context.

Rules:
- Resolve pronouns: "it", "those", "them", "that" → the specific entity from history.
- Resolve references: "same filter", "same status", "those cases" → repeat the exact condition.
- Resolve follow-ups: "what about X?" or "and Y?" → expand to the full question.
- If the message is already fully self-contained (no dependency on history), return it UNCHANGED.
- Return ONLY the rewritten question — no explanation, no prefix, no punctuation changes.

Conversation history (most recent last):
{history}"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for i, turn in enumerate(history, 1):
        lines.append(f"Turn {i}: User asked: {turn['query']}")
        if turn.get("summary"):
            lines.append(f"         Result: {turn['summary']}")
    return "\n".join(lines)


def _needs_resolution(query: str, history: list[dict]) -> bool:
    """Skip LLM call when there is no history or no reference words."""
    if not history:
        return False
    return bool(_REFERENCE_PATTERN.search(query))


def _extract_token_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input":  int(usage.get("input_tokens", 0)),
        "output": int(usage.get("output_tokens", 0)),
        "total":  int(usage.get("total_tokens", 0)),
    }


def _resolve_query(query: str, history: list[dict]) -> tuple[str, dict]:
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )
    system = SYSTEM_PROMPT.format(history=_format_history(history))
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
    content = response.content
    if isinstance(content, list):
        content = "".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
    return content.strip(), _extract_token_usage(response)


def context_loader_node(state: GraphState) -> GraphState:
    query = state["user_query"]
    history = state.get("conversation_history", [])

    # Story 6.2 — resolve role → permitted tables (fast, cached, no LLM)
    role = state.get("user_role") or "ES"
    permitted_tables = get_permitted_tables(role)
    role_display = get_role_display(role)

    token_usage = dict(state.get("token_usage") or {})
    if not _needs_resolution(query, history):
        resolved = query
        changed = False
    else:
        resolved, tok = _resolve_query(query, history)
        changed = resolved.lower().strip() != query.lower().strip()
        token_usage["Context Loader"] = tok

    # Story 4.2 — scan the resolved query for known glossary terms
    glossary_matches = find_matches(resolved)
    glossary_terms = [m["term"] for m in glossary_matches]

    detail = []
    if changed:
        detail += [f"Original: {query}", f"Resolved: {resolved}"]
    if glossary_terms:
        detail.append(f"Glossary terms detected: {', '.join(glossary_terms)}")

    if not _needs_resolution(query, history) and not changed:
        summary = (
            "No references detected — query is self-contained"
            if history
            else "First turn — no history yet"
        )
    elif changed:
        summary = "Query resolved using session history"
    else:
        summary = "Query unchanged after resolution check"

    if glossary_terms:
        summary += f" · {len(glossary_terms)} glossary term(s) matched"
    summary += f" · role: {role} ({len(permitted_tables)} tables)"

    detail.append(f"Role: {role_display} — {len(permitted_tables)} permitted tables")

    trace_entry = {
        "agent": "Context Loader",
        "status": "ok",
        "summary": summary,
        "detail": detail,
    }
    trace = state.get("agent_trace", []) + [trace_entry]
    return {
        **state,
        "resolved_query":   resolved,
        "glossary_matches": glossary_matches,
        "user_role":        role,
        "permitted_tables": permitted_tables,
        "agent_trace":      trace,
        "token_usage":      token_usage,
    }
