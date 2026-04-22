"""
Output Formatter Agent

Runs after the Executor and before the Response Formatter.
Automatically detects column types in the result rows and applies
human-readable formatting in-place, so all downstream agents
(Response Formatter, Feedback Store, CSV download) see clean data.

Detection strategy (in priority order):
  1. Value-type override — datetime → date, bool → raw, int → count/days.
     This ensures COUNT(*) integer results are never formatted as currency
     even if the column name contains "total" or "value".
  2. Column-name keyword matching (camelCase and snake_case aware).
  3. Decimal-with-scale fallback for numeric columns with non-integer values.

Formatting applied:
  - Currency → $1,234.56
  - Date     → Apr 2, 2026
  - Days     → "42 days"
  - Count    → 1,234,567 (comma-separated)
"""
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from state.context import GraphState


# ── Keyword tokenization (camelCase + snake_case aware) ──────────────────────

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")


def _tokenize(col_name: str) -> set:
    """
    Split a column name into lowercase tokens.
    Handles both camelCase (varianceAmount) and snake_case (nip_paid_amount).

    Examples:
      varianceAmount       → {variance, amount}
      nip_paid_amount      → {nip, paid, amount}
      createdAt            → {created, at}
      total_disputes       → {total, disputes}
      expected_total_fee   → {expected, total, fee}
    """
    # Convert camelCase boundaries into underscores first
    snake = _CAMEL_SPLIT.sub("_", col_name).lower()
    # Split on underscore, dash, or whitespace
    return {t for t in re.split(r"[_\-\s]+", snake) if t}


# Keyword sets — any token match = classification hit
_CURRENCY_KEYWORDS = {
    "amount", "balance", "payment", "fee", "revenue", "cost", "price", "value",
    "disbursement", "refund", "charge", "paid", "owed", "earned",
    "dollar", "dollars", "usd", "money", "total", "subtotal", "grandtotal",
    "variance", "allocated", "expected",
}

# Ambiguous — these alone don't trigger currency (e.g. "total_disputes").
# They do trigger if also paired with another currency keyword.
_AMBIG_CURRENCY = {"total", "value", "expected"}

_DATE_KEYWORDS = {
    "date", "time", "created", "updated", "closed", "opened", "filed",
    "submitted", "received", "rendered", "changed", "month", "year",
    "period", "timestamp", "due", "scheduled", "paidat",
}

# `at` and `on` alone are ambiguous — they only count as dates in suffix form
_DATE_SUFFIX_RE = re.compile(r"(?:_at|_on|At|On)$")

_DAYS_KEYWORDS = {"days", "duration", "elapsed", "turnaround", "lag"}

_PERCENTAGE_KEYWORDS = {"percent", "pct", "rate", "ratio", "proportion", "percentage", "win"}

# "number" excluded — appears in identifiers like dispute_number, invoice_number
_COUNT_KEYWORDS = {"count", "num", "qty", "quantity", "cnt", "total"}


# ── Value formatters ──────────────────────────────────────────────────────────

def _fmt_currency(val) -> str:
    try:
        if isinstance(val, Decimal):
            f = float(val)
        elif isinstance(val, str):
            f = float(Decimal(val))  # handles "0E-30" scientific notation
        else:
            f = float(val)
        return f"${f:,.2f}"
    except (TypeError, ValueError, InvalidOperation):
        return str(val)


def _fmt_date(val) -> str:
    def _clean(dt: datetime) -> str:
        day = dt.strftime("%d").lstrip("0")
        if dt.hour or dt.minute:
            return dt.strftime(f"%b {day}, %Y %H:%M")
        return dt.strftime(f"%b {day}, %Y")

    if isinstance(val, datetime):
        return _clean(val)
    if isinstance(val, date):
        day = val.strftime("%d").lstrip("0")
        return val.strftime(f"%b {day}, %Y")
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d", "%Y-%m"):
            try:
                dt = datetime.strptime(val.strip(), fmt)
                if fmt == "%Y-%m":
                    return dt.strftime("%b %Y")
                return _clean(dt)
            except ValueError:
                continue
    return str(val)


def _fmt_days(val) -> str:
    try:
        n = int(float(val))
        return f"{n:,} days"
    except (TypeError, ValueError):
        return str(val)


def _fmt_count(val) -> str:
    try:
        n = int(float(val))
        return f"{n:,}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_percentage(val) -> str:
    try:
        f = float(val)
        return f"{f:.2f}%"
    except (TypeError, ValueError):
        return str(val)


# ── Column type detection ─────────────────────────────────────────────────────

def _first_non_null(sample_vals: list):
    return next((v for v in sample_vals if v is not None), None)


def _name_says_currency(tokens: set, col_name: str) -> bool:
    hits = tokens & _CURRENCY_KEYWORDS
    if not hits:
        return False
    # Ambiguous tokens alone don't make a currency column.
    # e.g. "total_disputes" → {total, disputes}. "total" is ambiguous.
    if hits <= _AMBIG_CURRENCY:
        # Only ambiguous tokens matched; require a disambiguating hint
        # (value-type inspection later will decide).
        return False
    return True


def _name_says_date(tokens: set, col_name: str) -> bool:
    if tokens & _DATE_KEYWORDS:
        return True
    return bool(_DATE_SUFFIX_RE.search(col_name))


def _detect_col_type(col_name: str, sample_vals: list) -> str:
    """Return 'currency', 'date', 'days', 'count', or 'raw'."""
    tokens = _tokenize(col_name)
    first_val = _first_non_null(sample_vals)

    # ── 1. Value-type overrides ─────────────────────────────────────────────
    if first_val is not None:
        # Datetime wins over everything
        if isinstance(first_val, (date, datetime)):
            return "date"
        # Booleans are raw (isinstance(True, int) is True — check before int)
        if isinstance(first_val, bool):
            return "raw"
        # Plain int → count/days, regardless of misleading name
        if isinstance(first_val, int):
            if tokens & _DAYS_KEYWORDS:
                return "days"
            return "count"
        # String values → raw. Labels/descriptions/statuses often contain
        # currency-sounding words ("Payment Type", "Paid in full") but are
        # plain text, not numbers.
        if isinstance(first_val, str):
            # But a date-string in a date-named column should still parse as date
            if _name_says_date(tokens, col_name):
                return "date"
            return "raw"

    # ── 2. Decimal / float: numeric columns ────────────────────────────────
    numeric = isinstance(first_val, (Decimal, float))

    # Decimals with any fractional scale → currency
    if isinstance(first_val, Decimal):
        _, _, exponent = first_val.as_tuple()
        if isinstance(exponent, int) and exponent < 0:
            # Fractional — currency unless the name clearly says days
            if tokens & _DAYS_KEYWORDS:
                return "days"
            return "currency"

    # ── 3. Name-keyword matching for numeric-or-unknown values ─────────────
    if tokens & _PERCENTAGE_KEYWORDS:
        return "percentage"
    if _name_says_currency(tokens, col_name):
        return "currency"
    if _name_says_date(tokens, col_name):
        return "date"
    if tokens & _DAYS_KEYWORDS:
        return "days"
    # Ambiguous currency tokens (total, value, expected) on a numeric value → currency
    if numeric and tokens & _AMBIG_CURRENCY:
        return "currency"
    if tokens & _COUNT_KEYWORDS:
        return "count"

    return "raw"


# ── Main formatting logic ─────────────────────────────────────────────────────

def _format_rows(rows: list) -> list:
    """Return a new list of dicts with values formatted in-place."""
    if not rows:
        return rows

    cols = list(rows[0].keys())

    col_types = {}
    for col in cols:
        sample = [r[col] for r in rows[:10] if r.get(col) is not None]
        col_types[col] = _detect_col_type(col, sample)

    formatted = []
    for row in rows:
        new_row = {}
        for col in cols:
            val = row.get(col)
            if val is None:
                new_row[col] = None
                continue
            ctype = col_types[col]
            if ctype == "percentage":
                new_row[col] = _fmt_percentage(val)
            elif ctype == "currency":
                new_row[col] = _fmt_currency(val)
            elif ctype == "date":
                new_row[col] = _fmt_date(val)
            elif ctype == "days":
                new_row[col] = _fmt_days(val)
            elif ctype == "count":
                new_row[col] = _fmt_count(val)
            else:
                new_row[col] = val
        formatted.append(new_row)

    return formatted


# ── LangGraph node ────────────────────────────────────────────────────────────

def output_formatter_node(state: GraphState) -> GraphState:
    rows = state.get("query_result")
    if not rows:
        return state

    formatted = _format_rows(rows)

    col_types_summary = {}
    cols = list(rows[0].keys())
    for col in cols:
        sample = [r[col] for r in rows[:10] if r.get(col) is not None]
        col_types_summary[col] = _detect_col_type(col, sample)

    applied = [f"{col}→{t}" for col, t in col_types_summary.items() if t != "raw"]

    trace_entry = {
        "agent": "Output Formatter",
        "status": "ok",
        "summary": (
            f"Formatted {len(applied)} column(s)" if applied
            else "No formatting needed — all columns are plain values"
        ),
        "detail": applied if applied else [],
    }

    return {
        **state,
        "query_result": formatted,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }
