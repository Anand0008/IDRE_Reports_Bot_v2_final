"""
Executor Agent
Runs validated SQL against the database (read-only user).
Enforces row cap and timeout.
"""
import re
from sqlalchemy import text
from db.connector import get_engine
from state.context import GraphState

# Safety cap only — prevents a runaway query from OOM-ing the server.
# Normal result sets are returned in full; the UI shows first 50 and the
# full set is available via CSV download. Queries exceeding this cap are
# truncated and the user is informed — `execute_unlimited()` can fetch the
# complete result set if needed.
ROW_LIMIT = 50000
QUERY_TIMEOUT_SECONDS = 30


def _enforce_limit(sql: str) -> str:
    """Append a safety-cap LIMIT only if the SQL has no LIMIT and isn't a pure aggregate."""
    stripped = sql.strip().rstrip(";")
    if re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        return stripped
    # Skip limit on single-row aggregates (no GROUP BY)
    if re.search(r"\bCOUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\(", stripped, re.IGNORECASE):
        if not re.search(r"\bGROUP\s+BY\b", stripped, re.IGNORECASE):
            return stripped
    return f"{stripped} LIMIT {ROW_LIMIT}"


def execute_query(sql: str) -> tuple[list[dict], str]:
    """
    Execute SQL and return (rows, error).
    rows is a list of dicts. error is empty string on success.
    """
    sql = _enforce_limit(sql)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # MySQL: set statement timeout via session variable
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={QUERY_TIMEOUT_SECONDS * 1000}"))
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return rows, ""
    except Exception as e:
        return [], str(e)


DOWNLOAD_TIMEOUT_SECONDS = 120
CSV_ROW_LIMIT = 100_000  # higher cap for CSV downloads — 1 lakh rows


def execute_unlimited(sql: str) -> tuple[list[dict], str]:
    """
    Execute SQL with the higher CSV cap (100,000 rows) — used only for full
    CSV downloads. Appends LIMIT if none present; respects explicit LIMITs.
    2-minute timeout applies.
    """
    stripped = sql.strip().rstrip(";")
    # Apply the CSV cap unless the SQL already has an explicit LIMIT the user asked for
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        # Skip limit on single-row aggregates
        is_agg = (re.search(r"\bCOUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\(", stripped, re.IGNORECASE)
                  and not re.search(r"\bGROUP\s+BY\b", stripped, re.IGNORECASE))
        if not is_agg:
            stripped = f"{stripped} LIMIT {CSV_ROW_LIMIT}"

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={DOWNLOAD_TIMEOUT_SECONDS * 1000}"))
            result = conn.execute(text(stripped))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return rows, ""
    except Exception as e:
        return [], str(e)


def executor_node(state: GraphState) -> GraphState:
    sql = state.get("validated_sql", "")
    rows, error = execute_query(sql)
    trace = state.get("agent_trace", [])

    if error:
        trace_entry = {
            "agent": "Executor",
            "status": "error",
            "summary": "Query execution failed",
            "detail": [error[:200]],
        }
        trace = trace + [trace_entry]
        return {**state, "query_result": None, "row_count": 0, "execution_error": error, "agent_trace": trace}
    else:
        trace_entry = {
            "agent": "Executor",
            "status": "ok",
            "summary": f"Query executed successfully · {len(rows):,} row(s) returned",
            "detail": [],
        }
        trace = trace + [trace_entry]
        return {**state, "query_result": rows, "row_count": len(rows), "execution_error": None, "agent_trace": trace}
