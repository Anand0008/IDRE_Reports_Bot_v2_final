"""
Schema Verifier Agent (Improvement 1)

Runs SHOW COLUMNS FROM <table> for each table selected by Schema Mapper.
Builds a verified schema block with explicit negatives ("This table does NOT
have column X") to prevent the LLM from hallucinating columns.

Injected into the pipeline between platform_context and sql_writer.
"""
import re
from sqlalchemy import text
from db.connector import get_engine
from state.context import GraphState

_column_cache: dict[str, list[dict]] = {}


def _fetch_columns(table_name: str) -> list[dict]:
    if table_name in _column_cache:
        return _column_cache[table_name]
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`"))
            cols = []
            for row in result.fetchall():
                cols.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "key": row[3] or "",
                })
            _column_cache[table_name] = cols
            return cols
    except Exception:
        return []


_COMMONLY_HALLUCINATED = {
    "case": [
        "paymentType", "partyType", "disputeId", "caseNumber",
        "closed_date", "closedDate", "closedAt", "amount",
        "totalAmount", "payment_status", "arbitratorId",
    ],
    "payment": [
        "paymentType", "caseId", "partyType", "disputeId",
    ],
    "case_payment_allocation": [
        "amount", "paymentType", "status",
    ],
    "case_party": [
        "partyType_INITIATING", "partyType_NON_INITIATING",
        "role", "organizationId",
    ],
    "user": [
        "role",
    ],
}


def build_verified_schema(tables: list[str]) -> str:
    blocks = []
    for table in tables:
        cols = _fetch_columns(table)
        if not cols:
            continue

        col_names = {c["name"] for c in cols}
        col_lines = []
        for c in cols:
            key_info = f" [{c['key']}]" if c['key'] else ""
            null_info = " NULL" if c['nullable'] else " NOT NULL"
            col_lines.append(f"  - {c['name']} ({c['type']}{null_info}{key_info})")

        block = f"=== VERIFIED COLUMNS: `{table}` ===\n"
        block += "\n".join(col_lines)

        hallucinated = _COMMONLY_HALLUCINATED.get(table, [])
        negatives = [h for h in hallucinated if h not in col_names]
        if negatives:
            block += f"\n  WARNING: `{table}` does NOT have columns: {', '.join(negatives)}"
            block += "\n  DO NOT use these column names — they will cause runtime errors."

        blocks.append(block)

    if not blocks:
        return ""
    return "--- LIVE SCHEMA (verified via SHOW COLUMNS) ---\n\n" + "\n\n".join(blocks)


def schema_verifier_node(state: GraphState) -> GraphState:
    tables = state.get("relevant_tables", [])
    if not tables:
        return state

    verified = build_verified_schema(tables)
    if not verified:
        trace_entry = {
            "agent": "Schema Verifier",
            "status": "warn",
            "summary": "Could not verify schema — DB may be unreachable",
            "detail": [],
        }
        trace = state.get("agent_trace", []) + [trace_entry]
        return {**state, "agent_trace": trace}

    existing_ctx = state.get("schema_context", "")
    enriched = verified + "\n\n" + existing_ctx

    negatives_count = verified.count("does NOT have")
    trace_entry = {
        "agent": "Schema Verifier",
        "status": "ok",
        "summary": f"Verified columns for {len(tables)} table(s) · {negatives_count} hallucination warning(s)",
        "detail": [f"Tables verified: {', '.join(tables)}"],
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {**state, "schema_context": enriched, "agent_trace": trace}
