"""
Post-SQL Processor Agent — applies business logic computations after SQL execution.

This agent sits between the Executor and the Response Formatter. It takes raw
SQL results and enriches them with computed fields that the IDRE platform
calculates in application code (not SQL):

1. Business day counting (deadlines, due dates)
2. Due date urgency scoring (overdue/urgent/warning/normal)
3. Cents-to-dollars conversion (refundAmountCents → dollars)
4. Processing time calculation (days between statuses)
5. Historical comparison (period-over-period % change)
6. EST timezone awareness (daily report boundaries)
7. Pricing calculations (expected fees by dispute type)
8. Dispute number formatting (shortId → DISP-XXXXXXX)
9. Soft-delete filtering (remove REMOVED line items, deleted notes)
10. Payment variance detection (overpayment/underpayment)
11. "Paid in full" determination per party
"""
import re
import math
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional
from state.context import GraphState

# EST offset: UTC-5 (EST) / UTC-4 (EDT) — use fixed EST-5 for daily boundary calculations
_EST_OFFSET = timedelta(hours=-5)


# ── Pricing Constants (from IDRE lib/constants/pricing.ts) ─────────────────

PRICING = {
    "SINGLE":  {"entity_fee": 595.00, "cms_fee": 115.00, "total": 710.00},
    "BUNDLED": {"entity_fee": 595.00, "cms_fee": 115.00, "total": 710.00},
    "BATCHED": {"entity_fee": 795.00, "cms_fee": 115.00, "base_total": 910.00,
                "surcharge_per_25": 150.00},
}

TERMINAL_STATUSES = {
    "CLOSED_DEFAULT", "CLOSED_INITIATING_PARTY", "CLOSED_NON_INITIATING_PARTY",
    "CLOSED_ADMINISTRATIVE", "CLOSED_SPLIT_DECISION",
    "NOTICE_OF_DISMISSAL_NON_PAYMENT", "CLOSED_DEFAULT_IP", "CLOSED_DEFAULT_NIP",
    "INELIGIBLE",
}

REFUND_CENTS_COLUMNS = {"refundAmountCents", "refund_amount_cents"}


# ── Helper Functions ──────────────────────────────────────────────────────────

def _to_number(val: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", ""))
        except ValueError:
            return None
    return None


def _to_date(val: Any) -> Optional[datetime]:
    """Safely parse a date value."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime.combine(val, datetime.min.time())
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                     "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(val[:26], fmt)
            except (ValueError, IndexError):
                continue
    return None


def _add_business_days(start: datetime, days: int) -> datetime:
    """Add N business days to a date, skipping weekends."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current


def _business_days_between(start: datetime, end: datetime) -> int:
    """Count business days between two dates."""
    if start > end:
        start, end = end, start
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def _urgency_level(due_date: datetime, now: Optional[datetime] = None) -> dict:
    """Calculate urgency level for a due date."""
    now = now or datetime.now()
    diff = (due_date - now).days

    if diff < 0:
        return {"urgency": "overdue", "days": abs(diff),
                "message": f"{abs(diff)} day(s) overdue"}
    elif diff <= 1:
        return {"urgency": "urgent", "days": diff,
                "message": f"Due {'today' if diff == 0 else 'tomorrow'}"}
    elif diff <= 3:
        return {"urgency": "warning", "days": diff,
                "message": f"Due in {diff} days"}
    else:
        return {"urgency": "normal", "days": diff,
                "message": f"Due in {diff} days"}


def _expected_fee(dispute_type: str, line_item_count: int = 1) -> float:
    """Calculate expected per-party fee based on dispute type."""
    dtype = (dispute_type or "SINGLE").upper()
    if dtype == "BATCHED":
        base = PRICING["BATCHED"]["base_total"]
        if line_item_count > 25:
            import math
            extra_groups = math.ceil((line_item_count - 25) / 25)
            return base + extra_groups * PRICING["BATCHED"]["surcharge_per_25"]
        return base
    return PRICING.get(dtype, PRICING["SINGLE"])["total"]


# ── Processing Functions ──────────────────────────────────────────────────────

def _process_dispute_numbers(rows: list[dict]) -> list[dict]:
    """Format shortId as DISP-XXXXXXX for display."""
    for row in rows:
        if "dispute_number" in row and row["dispute_number"]:
            val = str(row["dispute_number"])
            if not val.startswith("DISP-"):
                row["dispute_number"] = f"DISP-{val}"
        elif "shortId" in row and row["shortId"]:
            row["dispute_number"] = f"DISP-{row['shortId']}"
    return rows


def _process_cents_to_dollars(rows: list[dict]) -> list[dict]:
    """Convert any *Cents columns to dollar amounts."""
    for row in rows:
        for key in list(row.keys()):
            if key.endswith("Cents") or key.endswith("_cents"):
                val = _to_number(row[key])
                if val is not None:
                    dollar_key = key.replace("Cents", "").replace("_cents", "") + "_dollars"
                    row[dollar_key] = round(val / 100, 2)
    return rows


def _process_urgency_scoring(rows: list[dict]) -> list[dict]:
    """Add urgency scoring for rows with due_date fields. Uses EST time for daily boundaries."""
    now = _now_est()
    for row in rows:
        for key in ["due_date", "dueDate", "primary_due_date", "eligibilityDueDate",
                     "paymentDueDate", "due_date_until_decision"]:
            if key in row and row[key]:
                dt = _to_date(row[key])
                if dt:
                    urgency = _urgency_level(dt, now)
                    row[f"{key}_urgency"] = urgency["urgency"]
                    row[f"{key}_message"] = urgency["message"]
                    row[f"{key}_days_remaining"] = urgency["days"]
    return rows


def _process_processing_time(rows: list[dict]) -> list[dict]:
    """Calculate processing time in days for rows with creation and status change dates."""
    for row in rows:
        created = _to_date(row.get("createdAt") or row.get("created_at"))
        changed = _to_date(row.get("statusChangedAt") or row.get("status_changed_at")
                           or row.get("closed_at"))
        if created and changed:
            diff = (changed - created).days
            row["processing_time_days"] = max(diff, 0)
    return rows


def _process_payment_status(rows: list[dict]) -> list[dict]:
    """Enrich payment rows with human-readable direction/type labels."""
    direction_labels = {"INCOMING": "Received", "OUTGOING": "Sent"}
    type_labels = {
        "CASE_PAYMENT": "Case Fee",
        "REFUND_TO_PREVAILING_PARTY": "Party Refund",
        "PARTY_REFUND_IP": "IP Refund",
        "PARTY_REFUND_NIP": "NIP Refund",
        "CAPITOL_BRIDGE_FEE": "Capitol Bridge Fee",
        "THIRD_PARTY_PAYMENT": "Internal Payout",
        "CMS_INVOICE_PAYMENT": "CMS Payment",
        "CMS_ADMIN_FEE_TRANSFER": "CMS Admin Fee",
    }
    for row in rows:
        if "direction" in row:
            row["direction_label"] = direction_labels.get(row["direction"], row["direction"])
        # payment.type is the actual column name (not paymentType)
        pay_type = row.get("type") or row.get("paymentType")
        if pay_type:
            row["payment_type_label"] = type_labels.get(pay_type, pay_type)
    return rows


def _process_soft_delete_filter(rows: list[dict]) -> list[dict]:
    """Remove soft-deleted rows (REMOVED line items, deleted notes)."""
    filtered = []
    for row in rows:
        # Skip removed line items
        if row.get("status") == "REMOVED" and "disputeName" in row:
            continue
        # Skip deleted notes
        if row.get("deletedAt") is not None and "case_note" in str(row.get("__table", "")):
            continue
        filtered.append(row)
    return filtered if filtered else rows  # Don't return empty if filter removes all


def _process_expected_fees(rows: list[dict]) -> list[dict]:
    """Add expected fee breakdown when disputeType column is present."""
    for row in rows:
        dtype = row.get("disputeType") or row.get("dispute_type") or row.get("typeOfDispute")
        if dtype:
            dtype_upper = str(dtype).upper()
            line_count = _to_number(row.get("line_item_count") or row.get("lineItemCount") or 1) or 1
            if dtype_upper == "BATCHED":
                base = PRICING["BATCHED"]["base_total"]
                if line_count > 25:
                    extra_groups = math.ceil((line_count - 25) / 25)
                    total = base + extra_groups * PRICING["BATCHED"]["surcharge_per_25"]
                else:
                    total = base
                row["expected_entity_fee"] = PRICING["BATCHED"]["entity_fee"]
                row["expected_cms_fee"] = PRICING["BATCHED"]["cms_fee"]
                row["expected_total_fee"] = total
            elif dtype_upper in PRICING:
                row["expected_entity_fee"] = PRICING[dtype_upper]["entity_fee"]
                row["expected_cms_fee"] = PRICING[dtype_upper]["cms_fee"]
                row["expected_total_fee"] = PRICING[dtype_upper]["total"]
    return rows


def _process_payment_variance(rows: list[dict]) -> list[dict]:
    """Add human-readable variance description for payment rows."""
    for row in rows:
        vtype = row.get("varianceType") or row.get("variance_type")
        vamount = _to_number(row.get("varianceAmount") or row.get("variance_amount"))
        if vtype:
            if vtype == "EXACT":
                row["variance_description"] = "Paid exact amount"
            elif vtype == "OVERPAYMENT" and vamount is not None:
                row["variance_description"] = f"Overpaid by ${abs(vamount):,.2f}"
            elif vtype == "UNDERPAYMENT" and vamount is not None:
                row["variance_description"] = f"Underpaid by ${abs(vamount):,.2f}"
            else:
                row["variance_description"] = vtype
    return rows


def _process_paid_in_full(rows: list[dict]) -> list[dict]:
    """Determine paid-in-full status when IP/NIP paid amounts are present."""
    for row in rows:
        ip_paid = _to_number(row.get("ip_paid_amount") or row.get("ip_total_paid"))
        nip_paid = _to_number(row.get("nip_paid_amount") or row.get("nip_total_paid"))
        dtype = str(row.get("disputeType") or row.get("typeOfDispute") or "SINGLE").upper()
        threshold = 910.0 if dtype == "BATCHED" else 710.0

        if ip_paid is not None:
            row["paid_in_full_ip"] = ip_paid >= threshold
            row["ip_payment_status"] = "Paid in full" if ip_paid >= threshold else f"${ip_paid:,.2f} of ${threshold:,.0f}"
        if nip_paid is not None:
            row["paid_in_full_nip"] = nip_paid >= threshold
            row["nip_payment_status"] = "Paid in full" if nip_paid >= threshold else f"${nip_paid:,.2f} of ${threshold:,.0f}"
        if ip_paid is not None and nip_paid is not None:
            row["both_paid_in_full"] = (ip_paid >= threshold) and (nip_paid >= threshold)
    return rows


def _process_historical_comparison(rows: list[dict]) -> list[dict]:
    """
    Calculate period-over-period % change for period-grouped result sets.
    Detects rows with a period/month/week column + numeric metric columns,
    then adds a pct_change column comparing each row to the previous.
    """
    if len(rows) < 2:
        return rows

    # Find the period column
    period_col = None
    for col in rows[0].keys():
        if re.search(r"\b(period|month|week|year|date|day)\b", col, re.IGNORECASE):
            period_col = col
            break
    if not period_col:
        return rows

    # Find numeric metric columns to compare
    metric_cols = []
    for col in rows[0].keys():
        if col == period_col:
            continue
        vals = [_to_number(r.get(col)) for r in rows if r.get(col) is not None]
        if vals and all(v is not None for v in vals[:3]):
            metric_cols.append(col)

    if not metric_cols:
        return rows

    # Add % change vs previous row for each metric column
    for i, row in enumerate(rows):
        if i == 0:
            for col in metric_cols:
                row[f"{col}_pct_change"] = None
        else:
            prev = rows[i - 1]
            for col in metric_cols:
                curr_val = _to_number(row.get(col))
                prev_val = _to_number(prev.get(col))
                if curr_val is not None and prev_val is not None and prev_val != 0:
                    pct = round(((curr_val - prev_val) / abs(prev_val)) * 100, 1)
                    row[f"{col}_pct_change"] = f"{'+' if pct >= 0 else ''}{pct}%"
                else:
                    row[f"{col}_pct_change"] = None

    return rows


def _now_est() -> datetime:
    """Return current datetime adjusted to EST (UTC-5)."""
    return datetime.utcnow() + _EST_OFFSET


# ── Main Post-Processor ──────────────────────────────────────────────────────

def _detect_needed_processors(query: str, sql: str, rows: list[dict]) -> list[str]:
    """Determine which post-processors to apply based on query, SQL, and result shape."""
    processors = []
    query_lower = (query or "").lower()
    sql_lower = (sql or "").lower()

    # Always format dispute numbers
    if rows and any("shortId" in r or "dispute_number" in r for r in rows[:5]):
        processors.append("dispute_numbers")

    # Cents-to-dollars if any cents columns present
    if rows and any(k.endswith("Cents") or k.endswith("_cents")
                     for r in rows[:5] for k in r.keys()):
        processors.append("cents_to_dollars")

    # Urgency scoring if due dates present
    if rows and any(k in r for r in rows[:5]
                     for k in ["due_date", "dueDate", "eligibilityDueDate",
                               "paymentDueDate", "due_date_until_decision"]):
        processors.append("urgency_scoring")

    # Processing time if both created and changed dates present
    if rows and any(("createdAt" in r or "created_at" in r) and
                     ("statusChangedAt" in r or "status_changed_at" in r or "closed_at" in r)
                     for r in rows[:5]):
        processors.append("processing_time")

    # Payment enrichment
    if rows and any("type" in r or "paymentType" in r or "direction" in r for r in rows[:5]):
        processors.append("payment_status")

    # Soft-delete filtering
    if "dispute_line_items" in sql_lower or "case_note" in sql_lower:
        processors.append("soft_delete_filter")

    # Expected fee calculation when disputeType present
    if rows and any("disputeType" in r or "dispute_type" in r or "typeOfDispute" in r for r in rows[:5]):
        processors.append("expected_fees")

    # Payment variance enrichment (handles both original and aliased column names)
    if rows and any("varianceType" in r or "variance_type" in r or "varianceAmount" in r or "variance_amount" in r for r in rows[:5]):
        processors.append("payment_variance")

    # Paid-in-full determination when IP/NIP payment amounts are in the result
    if rows and any(
        any(k in r for k in ("ip_paid_amount", "nip_paid_amount", "ip_total_paid", "nip_total_paid",
                             "initiating_paid", "non_initiating_paid"))
        for r in rows[:5]
    ):
        processors.append("paid_in_full")

    # Historical comparison for period-grouped results (2+ rows with period col)
    if (len(rows) >= 2 and
        any(re.search(r"\b(period|month|week|year)\b", k, re.IGNORECASE)
            for k in rows[0].keys())):
        processors.append("historical_comparison")

    return processors


def post_processor_node(state: GraphState) -> GraphState:
    """
    Apply post-SQL business logic computations to query results.
    This agent enriches raw SQL results with computed fields that
    the IDRE platform calculates in application code.
    """
    rows = state.get("query_result")
    if not rows or not isinstance(rows, list) or len(rows) == 0:
        trace_entry = {
            "agent": "Post-Processor",
            "status": "ok",
            "summary": "No rows to process — skipped",
            "detail": [],
        }
        trace = state.get("agent_trace", []) + [trace_entry]
        return {**state, "agent_trace": trace}

    query = state.get("resolved_query") or state.get("user_query", "")
    sql = state.get("validated_sql") or state.get("generated_sql", "")

    # Detect which processors are needed
    needed = _detect_needed_processors(query, sql, rows)

    # Apply processors
    applied = []
    for proc in needed:
        if proc == "dispute_numbers":
            rows = _process_dispute_numbers(rows)
            applied.append("Dispute number formatting (DISP-XXXXXXX)")
        elif proc == "cents_to_dollars":
            rows = _process_cents_to_dollars(rows)
            applied.append("Cents-to-dollars conversion")
        elif proc == "urgency_scoring":
            rows = _process_urgency_scoring(rows)
            applied.append("Due date urgency scoring")
        elif proc == "processing_time":
            rows = _process_processing_time(rows)
            applied.append("Processing time calculation (days)")
        elif proc == "payment_status":
            rows = _process_payment_status(rows)
            applied.append("Payment type/direction labels")
        elif proc == "soft_delete_filter":
            before = len(rows)
            rows = _process_soft_delete_filter(rows)
            after = len(rows)
            if before != after:
                applied.append(f"Soft-delete filter (removed {before - after} rows)")
            else:
                applied.append("Soft-delete filter (no rows removed)")
        elif proc == "expected_fees":
            rows = _process_expected_fees(rows)
            applied.append("Expected fee calculation (entity_fee/cms_fee/total_fee)")
        elif proc == "payment_variance":
            rows = _process_payment_variance(rows)
            applied.append("Payment variance enrichment (overpayment/underpayment labels)")
        elif proc == "paid_in_full":
            rows = _process_paid_in_full(rows)
            applied.append("Paid-in-full determination (IP/NIP vs $710/$910 threshold)")
        elif proc == "historical_comparison":
            rows = _process_historical_comparison(rows)
            applied.append("Period-over-period % change calculation")

    summary = f"Applied {len(applied)} post-processor(s)" if applied else "No post-processing needed"
    trace_entry = {
        "agent": "Post-Processor",
        "status": "ok",
        "summary": summary,
        "detail": applied if applied else ["All columns are raw SQL output — no enrichment needed"],
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {
        **state,
        "query_result": rows,
        "row_count": len(rows),
        "agent_trace": trace,
    }
