"""
Feedback Injector Agent

Activated only on feedback retry runs (is_feedback_retry=True).
Reads feedback_correction_context, formats it into a structured correction block,
and prepends it to retry_context so the SQL Writer acts on it automatically.

The SQL Writer already reads retry_context — this node uses that existing channel
with zero changes needed to sql_writer.py.
"""
from state.context import GraphState


def _format_correction_block(correction: dict) -> str:
    categories = correction.get("error_categories", [])
    note = correction.get("free_text_note", "").strip()
    original = correction.get("original_query", "").strip()
    summary = correction.get("original_result_summary", "").strip()

    lines = [
        "FEEDBACK CORRECTION (user-reported error):",
        f"Original question: {original}",
    ]

    if summary:
        lines.append(f"Previous answer shown: {summary[:200]}")

    if categories:
        lines.append("What the user reported was wrong:")
        for c in categories:
            lines.append(f"  - {c}")

    if note:
        lines.append(f'User note: "{note}"')

    lines += [
        "",
        "Instruction: The user has already seen one answer to this question and marked it "
        "as incorrect. Address each reported issue above directly and explicitly. "
        "Do NOT repeat the same approach as before. Treat this as a high-priority correction.",
    ]

    return "\n".join(lines)


def feedback_injector_node(state: GraphState) -> GraphState:
    correction = state.get("feedback_correction_context") or {}
    if not correction:
        return state

    correction_block = _format_correction_block(correction)

    # Prepend to retry_context so sql_writer picks it up through its existing channel
    existing = state.get("retry_context", "") or ""
    new_retry_context = correction_block + ("\n\n" + existing if existing else "")

    trace_entry = {
        "agent": "Feedback Injector",
        "status": "warn",
        "summary": "Correction context injected from user feedback",
        "detail": [f"Error categories: {correction.get('error_categories', [])}"],
    }

    return {
        **state,
        "retry_context": new_retry_context,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }
