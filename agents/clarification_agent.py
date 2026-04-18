"""
Clarification Agent  (Story 3.2)

Sits between ambiguity_scorer and schema_mapper.
If the ambiguity score is below THRESHOLD, or the query is a re-run after the
user already answered a clarification, it passes straight through.

Otherwise it calls Gemini to generate 1-2 short, targeted questions based on
the raised flags, writes them to clarification_question, and sets
needs_clarification=True so the orchestrator routes to END and the UI can
surface the question before running the rest of the pipeline.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings
from state.context import GraphState
from agents.ambiguity_scorer import _FLAG_BY_KEY

CLARIFICATION_THRESHOLD = 0.30   # score must exceed this to trigger a question

SYSTEM_PROMPT = """You are a helpful assistant for a dispute-resolution data platform.
A user asked a question that contains ambiguous or under-specified details.
Your job is to ask ONE or TWO short, plain-English clarifying questions to resolve the ambiguity.

Rules:
- Be specific: reference the exact ambiguous part of the query.
- Be concise: the entire response must be 1–3 sentences maximum.
- Give concrete options where possible (e.g. "last 7 days, 30 days, or this month?").
- Do NOT explain what you are doing — just ask the question(s).
- Do NOT use technical terms like "SQL", "filter", "schema", or "NULL".

Ambiguous query: {query}

Ambiguity flags raised:
{flag_details}"""


def _build_flag_details(flags: list[str]) -> str:
    lines = []
    for key in flags:
        flag = _FLAG_BY_KEY.get(key)
        if flag:
            lines.append(f"- {flag.label}: {flag.description}")
    return "\n".join(lines) if lines else "- General ambiguity"


def _extract_token_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input":  int(usage.get("input_tokens", 0)),
        "output": int(usage.get("output_tokens", 0)),
        "total":  int(usage.get("total_tokens", 0)),
    }


def _generate_clarification(query: str, flags: list[str]) -> tuple[str, dict]:
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.3,
        google_api_key=settings.gemini_api_key,
    )
    system = SYSTEM_PROMPT.format(
        query=query,
        flag_details=_build_flag_details(flags),
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content="Ask your clarifying question(s).")])
    content = response.content
    if isinstance(content, list):
        content = "".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
    return content.strip(), _extract_token_usage(response)


def clarification_agent_node(state: GraphState) -> GraphState:
    score    = state.get("ambiguity_score", 0.0)
    flags    = state.get("ambiguity_flags", [])
    query    = state.get("resolved_query") or state["user_query"]
    retried  = state.get("clarification_attempted", False)

    # Fast-pass conditions
    if retried or score <= CLARIFICATION_THRESHOLD or not flags:
        reason = (
            "Re-run after clarification — skipping check"
            if retried
            else f"Score {int(score * 100)}% ≤ threshold ({int(CLARIFICATION_THRESHOLD * 100)}%) — proceeding"
        )
        trace_entry = {
            "agent":   "Clarification Agent",
            "status":  "ok",
            "summary": reason,
            "detail":  [],
        }
        trace = state.get("agent_trace", []) + [trace_entry]
        return {
            **state,
            "needs_clarification":    False,
            "clarification_question": "",
            "agent_trace":            trace,
        }

    # Generate clarifying question(s)
    question, tok = _generate_clarification(query, flags)
    token_usage = dict(state.get("token_usage") or {})
    token_usage["Clarification Agent"] = tok

    trace_entry = {
        "agent":   "Clarification Agent",
        "status":  "warn",
        "summary": f"Score {int(score * 100)}% — pausing pipeline to ask for clarification",
        "detail":  [f"Flags: {', '.join(flags)}", f"Question: {question}"],
    }
    trace = state.get("agent_trace", []) + [trace_entry]
    return {
        **state,
        "needs_clarification":    True,
        "clarification_question": question,
        "agent_trace":            trace,
        "token_usage":            token_usage,
    }
