"""
SQL Validator Agent
Deterministic safety checks — no LLM involved.
Blocks DDL/DML, verifies tables exist, ensures query is SELECT-only.
Also validates column references against the schema catalog (Improvement 2).
"""
import re
import json
import os
from state.context import GraphState

SCHEMA_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "schema_catalog.json")
_catalog_cache = None

BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|CALL|GRANT|REVOKE|LOAD|OUTFILE|DUMPFILE)\b",
    re.IGNORECASE,
)


def _get_catalog() -> dict:
    global _catalog_cache
    if _catalog_cache is None:
        with open(SCHEMA_CATALOG_PATH) as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


def _get_known_tables() -> set:
    return set(_get_catalog()["tables"].keys())


def _get_table_columns(table_name: str) -> set:
    info = _get_catalog()["tables"].get(table_name, {})
    return {c["name"] for c in info.get("columns", [])}


def _extract_table_names(sql: str) -> list[str]:
    """Rough extraction of table names from FROM and JOIN clauses."""
    # Remove backtick quoting for matching
    normalized = sql.replace("`", "")
    pattern = re.compile(r"\b(?:FROM|JOIN)\s+(\w+)", re.IGNORECASE)
    return pattern.findall(normalized)


_TABLE_DOT_COL_RE = re.compile(
    r"`?(\w+)`?\s*\.\s*`?(\w+)`?",
)

_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+`?(\w+)`?\s+(?:AS\s+)?`?(\w+)`?",
    re.IGNORECASE,
)


def _extract_column_refs(sql: str, referenced_tables: list[str]) -> list[tuple[str, str]]:
    """Extract (table_or_alias, column) pairs from table.column references."""
    return _TABLE_DOT_COL_RE.findall(sql.replace("`", ""))


def _build_alias_map(sql: str) -> dict[str, str]:
    """Map aliases to real table names: {alias: table_name}."""
    alias_map = {}
    for match in _ALIAS_RE.finditer(sql.replace("`", "")):
        table, alias = match.group(1), match.group(2)
        if alias.upper() not in ("ON", "WHERE", "SET", "AND", "OR", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS"):
            alias_map[alias.lower()] = table.lower()
    return alias_map


def validate_columns(sql: str, referenced_tables: list[str]) -> list[str]:
    """Return list of warnings for hallucinated column references."""
    known_tables = _get_known_tables()
    alias_map = _build_alias_map(sql)
    refs = _extract_column_refs(sql, referenced_tables)
    warnings = []
    seen = set()

    for table_or_alias, col in refs:
        real_table = alias_map.get(table_or_alias.lower(), table_or_alias.lower())
        if real_table not in known_tables:
            continue
        key = (real_table, col)
        if key in seen:
            continue
        seen.add(key)
        valid_cols = _get_table_columns(real_table)
        if valid_cols and col not in valid_cols:
            warnings.append(f"Column `{real_table}`.`{col}` does not exist. Valid columns: {', '.join(sorted(valid_cols)[:15])}")

    return warnings


def validate_sql(sql: str, permitted_tables: list = None) -> tuple:
    """
    Returns (is_valid, error_message).
    error_message is empty string if valid.
    permitted_tables: if provided, any referenced table outside this list is blocked.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL generated."

    stripped = sql.strip()

    # Must start with SELECT
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False, f"Query must be a SELECT statement. Got: {stripped[:60]}"

    # Block dangerous keywords
    match = BLOCKED_KEYWORDS.search(stripped)
    if match:
        return False, f"Blocked keyword '{match.group()}' found in query."

    # Block multiple statements
    if stripped.rstrip(";").count(";") > 0:
        return False, "Multiple SQL statements are not allowed."

    # Check that referenced tables exist in the catalog
    known = _get_known_tables()
    referenced = _extract_table_names(stripped)
    unknown = [t for t in referenced if t not in known]
    if unknown:
        return False, f"Unknown table(s) referenced: {unknown}. Check spelling or schema."

    # Story 6.3 — permission re-check (defense-in-depth)
    if permitted_tables:
        unauthorized = [t for t in referenced if t not in permitted_tables]
        if unauthorized:
            # Do NOT reveal the table name — attacker should not learn what restricted tables exist
            return False, "Query references table(s) that are not accessible for your role."

    return True, ""


def sql_validator_node(state: GraphState) -> GraphState:
    sql = state.get("generated_sql", "")
    permitted = state.get("permitted_tables") or None
    is_valid, error = validate_sql(sql, permitted_tables=permitted)
    trace = state.get("agent_trace", [])

    if is_valid:
        tables_used = _extract_table_names(sql)
        col_warnings = validate_columns(sql, tables_used)

        if col_warnings:
            col_error = "Hallucinated column(s) detected:\n" + "\n".join(col_warnings)
            trace_entry = {
                "agent": "SQL Validator",
                "status": "error",
                "summary": f"Column validation failed — {len(col_warnings)} hallucinated column(s)",
                "detail": col_warnings,
            }
            trace = trace + [trace_entry]
            return {**state, "validated_sql": "", "error_message": col_error, "agent_trace": trace}

        trace_entry = {
            "agent": "SQL Validator",
            "status": "ok",
            "summary": f"Passed all safety checks · {len(tables_used)} table(s) referenced",
            "detail": [f"Tables in query: {', '.join(tables_used)}"] if tables_used else [],
        }
        trace = trace + [trace_entry]
        return {**state, "validated_sql": sql, "error_message": "", "agent_trace": trace}
    else:
        is_permission_violation = "not accessible for your role" in error
        trace_entry = {
            "agent": "SQL Validator",
            "status": "error",
            "summary": "Permission violation — query blocked" if is_permission_violation else "Validation failed — query blocked",
            "detail": [error],
        }
        trace = trace + [trace_entry]
        return {**state, "validated_sql": "", "error_message": error, "agent_trace": trace}
