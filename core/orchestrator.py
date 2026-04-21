"""
LangGraph Orchestrator (v2 — with Platform Context Agent)
Wires all agents into the pipeline with smart debugger-assisted retry logic.

Pipeline (normal):
  context_loader → ambiguity_scorer → clarification_agent
                                            │
                         needs_clarification=True → END  (question returned to UI)
                         needs_clarification=False ↓
                                       schema_mapper → platform_context → schema_verifier → sql_writer → sql_validator
                                                                                          │
                                                                             valid ───────┘
                                                                                          ↓
                                                                                       executor
                                                                                          │
                                                                             success ─────→ response_formatter → END
                                                                                          │
                                                                             error ───────→ debugger
                                                                                                │
                                                                             retries left ──────→ increment_retry → sql_writer
                                                                                                │
                                                                             max retries ───────→ max_retry_error → END

Feedback retry path:
  context_loader → feedback_injector → ambiguity_scorer → (normal path continues)
"""
from __future__ import annotations
import time
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from state.context import GraphState
from agents.context_loader import context_loader_node
from agents.feedback_injector import feedback_injector_node
from agents.ambiguity_scorer import ambiguity_scorer_node
from agents.clarification_agent import clarification_agent_node
from agents.schema_mapper import schema_mapper_node
from agents.platform_context_agent import platform_context_node
from agents.schema_verifier import schema_verifier_node
from agents.sql_writer import sql_writer_node
from agents.sql_validator import sql_validator_node
from agents.executor import executor_node
from agents.debugger_agent import debugger_node
from agents.post_processor import post_processor_node
from agents.response_formatter import response_formatter_node
from agents.output_formatter import output_formatter_node

MAX_RETRIES = 3
HISTORY_MAX_TURNS = 5


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_context_loader(
    state: GraphState,
) -> Literal["feedback_injector", "ambiguity_scorer"]:
    """Route through feedback_injector only on feedback retries — zero overhead otherwise."""
    if state.get("is_feedback_retry") and state.get("feedback_correction_context"):
        return "feedback_injector"
    return "ambiguity_scorer"


def _route_after_clarification(state: GraphState) -> Literal["schema_mapper", "__end__"]:
    if state.get("needs_clarification"):
        return "__end__"
    return "schema_mapper"


def _route_after_validator(
    state: GraphState,
) -> Literal["executor", "debugger", "__end__"]:
    if not state.get("error_message"):
        return "executor"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "debugger"
    return "__end__"


def _route_after_executor(
    state: GraphState,
) -> Literal["response_formatter", "debugger", "__end__"]:
    if not state.get("execution_error"):
        return "response_formatter"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "debugger"
    return "__end__"


def _route_after_debugger(
    state: GraphState,
) -> Literal["increment_retry", "max_retry_error"]:
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "increment_retry"
    return "max_retry_error"


# ── Helper nodes ───────────────────────────────────────────────────────────────

def _increment_retry(state: GraphState) -> GraphState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def _audit_trail(state: GraphState) -> GraphState:
    """Fire-and-forget async audit write at every terminal point."""
    try:
        from utils.audit_writer import build_and_log
        start_ms = state.get("pipeline_start_ms", 0)
        total_ms = int(time.time() * 1000) - start_ms if start_ms else 0
        build_and_log(state, total_ms)
    except Exception:
        pass
    return state


def _max_retry_error(state: GraphState) -> GraphState:
    """Graceful message after all retries are exhausted."""
    retry_ctx = state.get("retry_context", "")
    last_sql = state.get("generated_sql", "")
    raw_error = state.get("execution_error") or state.get("error_message") or "Unknown error"

    error_type = "unknown error"
    for line in retry_ctx.splitlines():
        if line.startswith("Error type"):
            error_type = line.split(":", 1)[-1].strip()
            break

    message_lines = [
        f"I was unable to generate a working query after {MAX_RETRIES} attempts.",
        f"",
        f"**Root cause:** {error_type}",
    ]

    if raw_error and error_type == "unknown error":
        message_lines.append(f"**Error detail:** {raw_error[:200]}")

    if last_sql:
        message_lines += [
            f"",
            f"**Last SQL attempted:**",
            f"```sql",
            last_sql[:400],
            f"```",
        ]

    message_lines += [
        f"",
        f"**Suggestions:**",
        f"- Try rephrasing your question with more specific column or status names",
        f"- Check that the table or column you're asking about exists in the database",
        f"- If asking about a payment count, specify 'completed payments' or 'P=0/P=1/P=2'",
        f"- Use exact status names like 'PENDING_PAYMENTS' or 'FINAL_DETERMINATION_RENDERED'",
        f"- For financial queries, specify payment direction (incoming/outgoing) and type",
    ]

    formatted = "\n".join(message_lines)

    trace_entry = {
        "agent": "Debugger",
        "status": "error",
        "summary": f"Max retries ({MAX_RETRIES}) exhausted — returning graceful error",
        "detail": [f"Error type: {error_type}"],
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {
        **state,
        "formatted_response": formatted,
        "agent_trace": trace,
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_pipeline():
    graph = StateGraph(GraphState)

    graph.add_node("context_loader",       context_loader_node)
    graph.add_node("feedback_injector",    feedback_injector_node)
    graph.add_node("ambiguity_scorer",     ambiguity_scorer_node)
    graph.add_node("clarification_agent",  clarification_agent_node)
    graph.add_node("schema_mapper",        schema_mapper_node)
    graph.add_node("platform_context",     platform_context_node)
    graph.add_node("schema_verifier",      schema_verifier_node)
    graph.add_node("sql_writer",           sql_writer_node)
    graph.add_node("sql_validator",        sql_validator_node)
    graph.add_node("executor",             executor_node)
    graph.add_node("post_processor",       post_processor_node)
    graph.add_node("debugger",             debugger_node)
    graph.add_node("increment_retry",      _increment_retry)
    graph.add_node("max_retry_error",      _max_retry_error)
    graph.add_node("response_formatter",   response_formatter_node)
    graph.add_node("output_formatter",     output_formatter_node)
    graph.add_node("audit_trail",          _audit_trail)

    graph.set_entry_point("context_loader")

    # Feedback retry path: context_loader → feedback_injector → ambiguity_scorer
    # Normal path:         context_loader → ambiguity_scorer
    graph.add_conditional_edges(
        "context_loader",
        _route_after_context_loader,
        {"feedback_injector": "feedback_injector", "ambiguity_scorer": "ambiguity_scorer"},
    )
    graph.add_edge("feedback_injector", "ambiguity_scorer")
    graph.add_edge("ambiguity_scorer",  "clarification_agent")

    graph.add_conditional_edges(
        "clarification_agent",
        _route_after_clarification,
        {"schema_mapper": "schema_mapper", "__end__": "audit_trail"},
    )

    # schema_mapper → platform_context → schema_verifier → sql_writer
    graph.add_edge("schema_mapper",      "platform_context")
    graph.add_edge("platform_context",   "schema_verifier")
    graph.add_edge("schema_verifier",    "sql_writer")
    graph.add_edge("sql_writer",        "sql_validator")

    graph.add_conditional_edges(
        "sql_validator",
        _route_after_validator,
        {"executor": "executor", "debugger": "debugger", "__end__": "audit_trail"},
    )

    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"response_formatter": "post_processor", "debugger": "debugger", "__end__": "audit_trail"},
    )

    graph.add_edge("post_processor",   "output_formatter")
    graph.add_edge("output_formatter", "response_formatter")

    graph.add_conditional_edges(
        "debugger",
        _route_after_debugger,
        {"increment_retry": "increment_retry", "max_retry_error": "max_retry_error"},
    )

    graph.add_edge("increment_retry",    "sql_writer")
    graph.add_edge("max_retry_error",    "audit_trail")
    graph.add_edge("response_formatter", "audit_trail")
    graph.add_edge("audit_trail",        END)

    return graph.compile()


pipeline = build_pipeline()


def run_query(
    user_query: str,
    session_id: str = "default",
    conversation_history: list[dict] = None,
    clarification_attempted: bool = False,
    user_role: str = "VO",
    user_identity: str = "",
    feedback_correction_context: dict = None,
    is_feedback_retry: bool = False,
) -> dict:
    """
    Run the full pipeline for a user query.

    clarification_attempted: set True when re-running after the user answered a
                             clarification question — prevents re-clarifying.
    user_identity:           display handle from post-auth prompt, tagged to audit trail.
    feedback_correction_context: structured correction from a "No" feedback submission.
    is_feedback_retry:       True when this run was triggered by feedback correction.
    """
    from utils.query_cache import get_cached, put_cached
    from agents.context_loader import _needs_resolution

    history = (conversation_history or [])[-HISTORY_MAX_TURNS:]

    # ── Cache check (skip for clarification re-runs and feedback retries) ─────
    if not clarification_attempted and not is_feedback_retry:
        if not _needs_resolution(user_query, history):
            resolved_for_cache = user_query
        else:
            resolved_for_cache = None

        if resolved_for_cache is not None:
            cached = get_cached(resolved_for_cache, user_role)
            if cached is not None:
                import copy
                result = copy.copy(cached)
                result["_from_cache"] = True
                result["agent_trace"] = [{
                    "agent": "Cache",
                    "status": "ok",
                    "summary": "Result served from cache — no LLM or DB calls needed",
                    "detail": [f"Cache key: {resolved_for_cache[:80]}"],
                }] + list(cached.get("agent_trace", []))
                return result

    initial_state: GraphState = {
        "user_query":                  user_query,
        "session_id":                  session_id,
        "user_role":                   user_role,
        "user_identity":               user_identity,
        "permitted_tables":            [],
        "conversation_history":        history,
        "resolved_query":              "",
        "glossary_matches":            [],
        "ambiguity_score":             0.0,
        "ambiguity_flags":             [],
        "needs_clarification":         False,
        "clarification_question":      "",
        "clarification_attempted":     clarification_attempted,
        "relevant_tables":             [],
        "schema_context":              "",
        "platform_context":            "",
        "generated_sql":               "",
        "validated_sql":               "",
        "query_result":                None,
        "row_count":                   0,
        "execution_error":             None,
        "formatted_response":          "",
        "assumptions":                 [],
        "response_format":             "table",
        "chart_config":                None,
        "query_explanation":           "",
        "proactive_suggestions":       [],
        "retry_count":                 0,
        "retry_context":               "",
        "error_message":               "",
        "pipeline_start_ms":           int(time.time() * 1000),
        "agent_timings":               {},
        "agent_trace":                 [],
        "token_usage":                 {},
        "feedback_correction_context": feedback_correction_context,
        "is_feedback_retry":           is_feedback_retry,
        "feedback_record_id":          (feedback_correction_context or {}).get("feedback_record_id", ""),
    }
    result = pipeline.invoke(initial_state)

    # ── Populate cache only for clean, non-retry results ─────────────────────
    if (
        result.get("formatted_response")
        and not result.get("needs_clarification")
        and not result.get("error_message")
        and not result.get("execution_error")
        and not is_feedback_retry
        and result.get("resolved_query")
    ):
        put_cached(result["resolved_query"], user_role, result)

    return result
