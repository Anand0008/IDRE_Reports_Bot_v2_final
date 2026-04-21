"""
Debugger Agent  (Story 5.1 — Smart Error Classification & Retry Context)

Sits between the Executor/Validator and the SQL Writer retry.
Classifies the raw error message into a typed category, extracts the
failing element (column name, table name, etc.), and generates a
targeted retry instruction for the SQL Writer.

Fully deterministic — no LLM call. Uses regex against known MySQL /
SQLAlchemy / Validator error patterns.

Error taxonomy
--------------
HALLUCINATED_COLUMN   — model invented a column that doesn't exist
HALLUCINATED_TABLE    — model invented or misspelled a table name
AMBIGUOUS_COLUMN      — column referenced without table qualifier
SYNTAX_ERROR          — SQL is malformed / wrong dialect
TIMEOUT               — query exceeded MAX_EXECUTION_TIME
DIVISION_BY_ZERO      — missing NULLIF guard
TYPE_MISMATCH         — wrong data type for a column / date format
SAFETY_VIOLATION      — DDL/DML keyword slipped through
WRONG_STATEMENT_TYPE  — not a SELECT
LOCK_TIMEOUT          — transient lock contention
UNKNOWN               — no pattern matched
"""
import re
from typing import NamedTuple
from state.context import GraphState

MAX_RETRIES = 3


class DebugResult(NamedTuple):
    error_type: str
    failing_element: str   # column name, table name, keyword, or ""
    retry_instruction: str  # targeted guidance string for the SQL Writer


# ── Pattern registry ──────────────────────────────────────────────────────────
# Each entry: (compiled_pattern, error_type, instruction_template)
# Use {element} in the template — it will be replaced with the captured group.

_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"Unknown column ['\"]?([^'\"]+)['\"]?", re.IGNORECASE),
        "HALLUCINATED_COLUMN",
        "Column '{element}' does not exist. Re-examine the schema context and use only "
        "columns listed there. Do not invent column names.",
    ),
    (
        re.compile(r"Table ['\"]?[^'\"]*\.['\"]?([^'\"]+)['\"]? doesn't exist", re.IGNORECASE),
        "HALLUCINATED_TABLE",
        "Table '{element}' does not exist in the database. Use only tables listed in the "
        "schema context. Check spelling carefully.",
    ),
    (
        re.compile(r"Unknown table\(s\) referenced[:\s]+\[([^\]]+)\]", re.IGNORECASE),
        "HALLUCINATED_TABLE",
        "Table(s) {element} do not exist. Use only the exact table names in the schema "
        "context — no invented or abbreviated names.",
    ),
    (
        re.compile(r"Hallucinated column\(s\) detected.*?Column `(\w+)`\.`(\w+)` does not exist", re.IGNORECASE | re.DOTALL),
        "HALLUCINATED_COLUMN",
        "The SQL Writer used column '{element}' which does not exist. "
        "Check the VERIFIED COLUMNS block in the schema context for the actual column names. "
        "Do NOT guess or fabricate column names — use only verified columns.",
    ),
    (
        re.compile(r"Column ['\"]?([^'\"]+)['\"]? in (?:field list|where clause|order clause|group statement) is ambiguous", re.IGNORECASE),
        "AMBIGUOUS_COLUMN",
        "Column '{element}' is ambiguous — it exists in multiple tables in the query. "
        "Qualify every column reference with its table name (e.g., `case`.`{element}`).",
    ),
    (
        re.compile(r"You have an error in your SQL syntax", re.IGNORECASE),
        "SYNTAX_ERROR",
        "The SQL has a syntax error. Verify: (1) all table names are backtick-quoted "
        "(especially `case`), (2) commas and parentheses are balanced, "
        "(3) subqueries are properly closed, (4) no trailing commas before FROM/WHERE.",
    ),
    (
        re.compile(r"Query execution was interrupted.*maximum statement execution time exceeded", re.IGNORECASE),
        "TIMEOUT",
        "The query timed out (exceeded 30 s). Rewrite with a more restrictive WHERE clause "
        "to reduce the scanned rows. Consider: (1) adding a date range filter on createdAt, "
        "(2) avoiding full-table scans, (3) replacing correlated subqueries with JOINs.",
    ),
    (
        re.compile(r"Division by zero", re.IGNORECASE),
        "DIVISION_BY_ZERO",
        "Division by zero detected. Wrap the denominator in NULLIF(expr, 0) so that "
        "zero denominators return NULL instead of erroring.",
    ),
    (
        re.compile(r"Incorrect (?:datetime|date|time) value[:\s]+['\"]?([^'\"]+)['\"]?", re.IGNORECASE),
        "TYPE_MISMATCH",
        "Invalid date/time value '{element}'. Use MySQL date functions: CURDATE(), NOW(), "
        "DATE_FORMAT(), DATE_SUB(). Do not use string literals for dates unless the column "
        "type is VARCHAR. Example: WHERE createdAt >= DATE_FORMAT(CURDATE(), '%Y-%m-01').",
    ),
    (
        re.compile(r"Incorrect integer value|Truncated incorrect|Data too long|Out of range value", re.IGNORECASE),
        "TYPE_MISMATCH",
        "Data type mismatch. Re-check the column type in the schema context and ensure the "
        "value being compared or inserted matches that type.",
    ),
    (
        re.compile(r"Blocked keyword ['\"]?(\w+)['\"]?", re.IGNORECASE),
        "SAFETY_VIOLATION",
        "The keyword '{element}' is not allowed. Only SELECT statements are permitted. "
        "Remove all DDL/DML keywords.",
    ),
    (
        re.compile(r"Query must be a SELECT statement", re.IGNORECASE),
        "WRONG_STATEMENT_TYPE",
        "Only SELECT queries are allowed. Rewrite as a SELECT statement.",
    ),
    (
        re.compile(r"Lock wait timeout exceeded", re.IGNORECASE),
        "LOCK_TIMEOUT",
        "A lock timeout occurred (transient). Simplify the query or reduce the number of "
        "rows it touches to avoid lock contention.",
    ),
]


def _classify(error: str) -> DebugResult:
    for pattern, error_type, template in _PATTERNS:
        m = pattern.search(error)
        if m:
            element = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else ""
            instruction = template.replace("{element}", element)
            return DebugResult(error_type, element, instruction)
    return DebugResult("UNKNOWN", "", f"Unclassified error: {error[:200]}")


def _build_retry_context(
    error: str,
    result: DebugResult,
    attempt: int,
    original_sql: str,
) -> str:
    """
    Compose the full retry context string injected into the SQL Writer prompt.
    """
    lines = [
        f"RETRY ATTEMPT {attempt} — ERROR ANALYSIS:",
        f"Error type   : {result.error_type}",
    ]
    if result.failing_element:
        lines.append(f"Failing item : {result.failing_element}")
    lines += [
        f"Raw error    : {error[:300]}",
        "",
        f"Fix instruction: {result.retry_instruction}",
        "",
        "Previous SQL that failed:",
        original_sql[:600],
    ]
    return "\n".join(lines)


# ── LangGraph node ────────────────────────────────────────────────────────────

def debugger_node(state: GraphState) -> GraphState:
    error = state.get("execution_error") or state.get("error_message") or "Unknown error"
    original_sql = state.get("validated_sql") or state.get("generated_sql") or ""
    retry_count = state.get("retry_count", 0)

    result = _classify(error)
    retry_ctx = _build_retry_context(error, result, retry_count + 1, original_sql)

    trace_entry = {
        "agent": "Debugger",
        "status": "warn",
        "summary": f"Error classified as {result.error_type}"
        + (f" · failing item: '{result.failing_element}'" if result.failing_element else ""),
        "detail": [result.retry_instruction],
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {
        **state,
        "retry_context": retry_ctx,
        "agent_trace": trace,
        # Clear the previous error so the SQL Writer sees a clean slate
        "execution_error": None,
        "error_message": "",
    }
