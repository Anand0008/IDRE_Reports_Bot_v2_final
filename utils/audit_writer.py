"""
Audit Trail Writer  (Story 8.1 — Async Audit Logging)

Writes AuditEvent records to data/audit_log.jsonl (newline-delimited JSON).
Each record is one JSON object per line — easy to tail, grep, or load into pandas.

All writes happen in a background thread so the pipeline never blocks on I/O.
If the write fails (disk full, permission error, etc.) the error is swallowed
silently — audit logging must never degrade the user-facing response.

Execution status vocabulary
---------------------------
success             — SQL ran, results returned
clarification       — pipeline paused to ask user a question
validation_failed   — SQL Validator blocked the query
permission_denied   — SQL referenced tables outside permitted_tables
execution_error     — DB returned an error (after all retries)
max_retries_exceeded— Debugger exhausted MAX_RETRIES
"""
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.jsonl")

_write_lock = threading.Lock()
_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class AuditEvent:
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    user_role: str = "ES"
    user_identity: str = ""
    timestamp_ist: str = field(
        default_factory=lambda: datetime.now(_IST).isoformat()
    )

    # Input
    user_query: str = ""
    resolved_query: str = ""
    ambiguity_score: float = 0.0
    glossary_terms_matched: list[str] = field(default_factory=list)

    # Schema
    relevant_tables: list[str] = field(default_factory=list)

    # SQL
    generated_sql: str = ""
    validated_sql: str = ""

    # Execution outcome
    execution_status: str = "success"   # see vocabulary above
    execution_error: str = ""
    retry_count: int = 0
    row_count: int = 0
    response_format: str = "table"

    # Timing (ms)
    total_pipeline_ms: int = 0
    agent_timings: dict[str, int] = field(default_factory=dict)

    # Flags
    clarification_asked: bool = False
    permission_violation: bool = False
    assumptions_count: int = 0
    is_feedback_retry: bool = False
    feedback_record_id: str = ""


def _write_sync(event: AuditEvent) -> None:
    """Synchronous write — called from a background thread."""
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    record = asdict(event)
    line = json.dumps(record, default=str) + "\n"
    with _write_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)


def log_event(event: AuditEvent) -> None:
    """Fire-and-forget: write the audit event in a background thread."""
    t = threading.Thread(target=_write_sync, args=(event,), daemon=True)
    t.start()


def _derive_status(state: dict[str, Any]) -> str:
    """Infer execution_status from the final pipeline state."""
    if state.get("needs_clarification"):
        return "clarification"
    if state.get("permission_violation"):
        return "permission_denied"
    err = state.get("error_message", "")
    if err and "not accessible for your role" in err:
        return "permission_denied"
    if err:
        return "validation_failed"
    if state.get("execution_error"):
        retry = state.get("retry_count", 0)
        from core.orchestrator import MAX_RETRIES
        return "max_retries_exceeded" if retry >= MAX_RETRIES else "execution_error"
    return "success"


def build_and_log(state: dict[str, Any], total_ms: int) -> None:
    """
    Convenience: construct an AuditEvent from the final GraphState and log it.
    Safe to call from any context — all exceptions are caught.
    """
    try:
        glossary_terms = [m["term"] for m in state.get("glossary_matches", [])]
        status = _derive_status(state)

        event = AuditEvent(
            session_id=state.get("session_id", ""),
            user_role=state.get("user_role", "ES"),
            user_identity=state.get("user_identity", ""),
            user_query=state.get("user_query", ""),
            resolved_query=state.get("resolved_query", ""),
            ambiguity_score=float(state.get("ambiguity_score", 0.0)),
            glossary_terms_matched=glossary_terms,
            relevant_tables=state.get("relevant_tables", []),
            generated_sql=state.get("generated_sql", ""),
            validated_sql=state.get("validated_sql", ""),
            execution_status=status,
            execution_error=str(state.get("execution_error") or state.get("error_message") or ""),
            retry_count=int(state.get("retry_count", 0)),
            row_count=int(state.get("row_count", 0)),
            response_format=state.get("response_format", "table"),
            total_pipeline_ms=total_ms,
            agent_timings=state.get("agent_timings", {}),
            clarification_asked=bool(state.get("needs_clarification")),
            permission_violation="not accessible for your role" in str(state.get("error_message", "")),
            assumptions_count=len(state.get("assumptions", [])),
            is_feedback_retry=bool(state.get("is_feedback_retry")),
            feedback_record_id=state.get("feedback_record_id", ""),
        )
        log_event(event)
    except Exception:
        pass  # audit failure must never surface to user
