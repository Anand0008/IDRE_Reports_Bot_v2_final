"""
Ambiguity Scorer Agent  (Story 3.1 + Improvement 4)

Pure pattern-based scorer — no LLM call.
Evaluates the resolved query for signals that it is under-specified.

Each flag has a weight; final score = sum of triggered weights, capped at 1.0.
Score and flags are written to GraphState for use by Story 3.2 (Clarification Agent).

Improvement 4: Default resolutions suppress flags for queries where glossary terms
provide sufficient context (e.g. "recent cases" → defaults to last 30 days).
"""
import re
from dataclasses import dataclass, field
from state.context import GraphState


# ── Flag definitions ──────────────────────────────────────────────────────────

@dataclass
class Flag:
    key: str
    label: str          # human-readable name shown in trace / UI
    description: str    # one-line explanation of what was detected
    weight: float       # contribution to overall ambiguity score


_FLAGS: list[Flag] = [
    Flag(
        key="vague_time",
        label="Vague time reference",
        description="No specific period — e.g. 'recent', 'lately', 'current' without a date range",
        weight=0.35,
    ),
    Flag(
        key="vague_quantity",
        label="Vague quantity",
        description="Imprecise quantity word used as a filter — e.g. 'many', 'few', 'several'",
        weight=0.20,
    ),
    Flag(
        key="unresolved_pronoun",
        label="Unresolved pronoun",
        description="Pronoun with no prior conversation history to resolve against",
        weight=0.45,
    ),
    Flag(
        key="broad_entity",
        label="Broad entity — no filter",
        description="Mentions cases/payments/disputes with no status, date, amount, or org qualifier",
        weight=0.25,
    ),
    Flag(
        key="ambiguous_metric",
        label="Ambiguous metric",
        description="'Amount' or 'total' used without specifying paid, pending, requested, or approved",
        weight=0.45,
    ),
    Flag(
        key="incomplete_range",
        label="Incomplete date range",
        description="Range expression is missing its end (e.g. 'between X and ?', 'from X to ?')",
        weight=0.60,
    ),
]

_FLAG_BY_KEY = {f.key: f for f in _FLAGS}


# ── Patterns ──────────────────────────────────────────────────────────────────

# Vague time words (but NOT "this month / last month / this week / last year" etc.)
_VAGUE_TIME_RE = re.compile(
    r"\b(recent(ly)?|lately|just|not long ago|a while(?: ago)?|"
    r"past few|last few|these days|soon|sometime|at some point|"
    r"currently|presently|nowadays)\b",
    re.IGNORECASE,
)

# "current" only counts as vague when NOT followed by a specific unit
_CURRENT_VAGUE_RE = re.compile(
    r"\bcurrent\b(?!\s+(?:month|week|year|quarter|day|date|fiscal))",
    re.IGNORECASE,
)

# Vague quantity — only flag when NOT preceded by "how" (avoids "how many")
_VAGUE_QTY_RE = re.compile(
    r"(?<!how )\b(many|few|several|a lot(?: of)?|handful(?: of)?|"
    r"numerous|a number of|some(?! of the)|a few)\b",
    re.IGNORECASE,
)

# Pronouns that need prior context
_PRONOUN_RE = re.compile(
    r"\b(it|its|they|them|their|those|these|that|such cases?|"
    r"the same|same ones?|those ones?)\b",
    re.IGNORECASE,
)

# Entity words that suggest a broad data scan
_ENTITY_RE = re.compile(
    r"\b(cases?|payments?|disputes?|invoices?|refunds?)\b",
    re.IGNORECASE,
)

# Qualifiers that narrow the scope — if any are present, suppress broad_entity flag
_QUALIFIER_RE = re.compile(
    r"\b(status|paid|unpaid|pending|closed|open|ineligible|eligible|"
    r"defaulted?|arbitrat|month|week|year|day|date|amount|total|"
    r"organisation|organization|org|insurer|provider|name|"
    r"greater|less|more than|at least|before|after|between|since|"
    r"mtd|ytd|q[1-4])\b"
    r"|[A-Z]{2,}(?:_[A-Z]+)+",   # ALL_CAPS_STATUS strings like PENDING_PAYMENTS
    re.IGNORECASE,
)

# Ambiguous metric — "amount(s)" / "payment amount" without a type qualifier
_AMBIGUOUS_METRIC_RE = re.compile(
    r"\b(payment\s+amounts?|amounts?\s+(?:of|for|by)|sum\s+of|total\s+(?:payment|amount))\b"
    r"|\bamounts?\b",
    re.IGNORECASE,
)
# If any of these appear alongside "amount", it's specific enough — suppress flag
_METRIC_QUALIFIER_RE = re.compile(
    r"\b(paid|unpaid|pending|approved|requested|refunded|allocated|settled|disbursed)\b",
    re.IGNORECASE,
)

# Incomplete range — "between X and ?" or trailing open-ended range
_INCOMPLETE_RANGE_RE = re.compile(
    r"\band\s+\?"             # "and ?" literal
    r"|\bto\s+\?"             # "to ?" literal
    r"|\bfrom\s+\S.*\bto\s*$" # "from X to" with nothing after
    r"|\bbetween\b.*\band\s*$",# "between X and" with nothing after
    re.IGNORECASE,
)


# ── Default resolutions (Improvement 4) ──────────────────────────────────────
# When these patterns match alongside an ambiguity flag, suppress that flag
# because the query has enough context for a reasonable default.

_DEFAULT_RESOLUTIONS: dict[str, list[re.Pattern]] = {
    "vague_time": [
        re.compile(r"\b(cases?|disputes?|payments?)\b", re.IGNORECASE),
    ],
    "broad_entity": [
        re.compile(r"\bhow many\b", re.IGNORECASE),
        re.compile(r"\btotal\b", re.IGNORECASE),
        re.compile(r"\bcount\b", re.IGNORECASE),
        re.compile(r"\blist\b", re.IGNORECASE),
        re.compile(r"\bshow\b", re.IGNORECASE),
        re.compile(r"\ball\b", re.IGNORECASE),
    ],
    "ambiguous_metric": [
        re.compile(r"\b(revenue|fees?|paid|collected|refund)\b", re.IGNORECASE),
    ],
}

_VAGUE_TIME_SUPPRESS = re.compile(
    r"\b(this month|last month|this week|last week|this year|last year|"
    r"today|yesterday|mtd|ytd|q[1-4]|20\d{2}|january|february|march|"
    r"april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


def _should_suppress(flag_key: str, query: str, glossary_matches: list) -> bool:
    """Check if a flag should be suppressed due to default resolution context."""
    if flag_key == "vague_time" and _VAGUE_TIME_SUPPRESS.search(query):
        return True

    if glossary_matches:
        for match in glossary_matches:
            cat = match.get("category", "")
            if flag_key == "vague_time" and cat == "time_range":
                return True
            if flag_key == "broad_entity" and cat in ("case_status", "payment_status", "payment_type", "dispute_type"):
                return True
            if flag_key == "ambiguous_metric" and cat in ("payment_type", "payment_status"):
                return True

    patterns = _DEFAULT_RESOLUTIONS.get(flag_key, [])
    for pattern in patterns:
        if pattern.search(query):
            return True

    return False


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_ambiguity(query: str, conversation_history: list[dict],
                    glossary_matches: list = None) -> tuple[float, list[str]]:
    """
    Returns (score, triggered_flag_keys).
    score is in [0.0, 1.0].  Higher = more ambiguous.
    """
    glossary_matches = glossary_matches or []
    triggered: list[str] = []

    # vague_time
    if _VAGUE_TIME_RE.search(query) or _CURRENT_VAGUE_RE.search(query):
        triggered.append("vague_time")

    # vague_quantity
    if _VAGUE_QTY_RE.search(query):
        triggered.append("vague_quantity")

    # unresolved_pronoun — only when there is no history to resolve against
    if _PRONOUN_RE.search(query) and not conversation_history:
        triggered.append("unresolved_pronoun")

    # broad_entity — entity word present but no narrowing qualifier
    if _ENTITY_RE.search(query) and not _QUALIFIER_RE.search(query):
        triggered.append("broad_entity")

    # ambiguous_metric — "amount" mentioned without specifying paid/pending/requested
    if _AMBIGUOUS_METRIC_RE.search(query) and not _METRIC_QUALIFIER_RE.search(query):
        triggered.append("ambiguous_metric")

    # incomplete_range — open-ended range like "between X and ?"
    if _INCOMPLETE_RANGE_RE.search(query):
        triggered.append("incomplete_range")

    # Suppress flags that have default resolutions
    triggered = [f for f in triggered if not _should_suppress(f, query, glossary_matches)]

    score = min(sum(_FLAG_BY_KEY[k].weight for k in triggered), 1.0)
    return round(score, 2), triggered


def ambiguity_scorer_node(state: GraphState) -> GraphState:
    query   = state.get("resolved_query") or state["user_query"]
    history = state.get("conversation_history", [])
    glossary = state.get("glossary_matches", [])

    score, flags = score_ambiguity(query, history, glossary_matches=glossary)

    if not flags:
        summary = "Query is unambiguous — no flags raised"
        status  = "ok"
        detail  = []
    else:
        pct     = int(score * 100)
        summary = f"Ambiguity score: {pct}% · {len(flags)} flag(s) raised"
        status  = "warn" if score < 0.6 else "error"
        detail  = [
            f"{_FLAG_BY_KEY[k].label}: {_FLAG_BY_KEY[k].description}"
            for k in flags
        ]

    trace_entry = {
        "agent":   "Ambiguity Scorer",
        "status":  status,
        "summary": summary,
        "detail":  detail,
    }
    trace = state.get("agent_trace", []) + [trace_entry]

    return {
        **state,
        "ambiguity_score": score,
        "ambiguity_flags": flags,
        "agent_trace":     trace,
    }
